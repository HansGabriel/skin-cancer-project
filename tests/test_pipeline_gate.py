"""End-to-end gate behaviour: what stops a scan, and what the user is told.

Three different failures must produce three different messages. The old
pipeline collapsed all of them into "image not clear", which is unhelpful when
the real problem is that the camera is pointed at a wall.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.contracts import ScanResult
from services.pipeline import run_pipeline

SIZE = 320
SKIN = (198, 134, 66)
LESION = (44, 28, 22)


class _Backend:
    """Always returns a confident benign — so anything that reaches it shows up."""

    backend_id = "mock"

    def __init__(self) -> None:
        self.calls = 0

    def scan(self, image_jpg_bytes=None) -> ScanResult:
        self.calls += 1
        return ScanResult(
            label="benign",
            confidence=95.0,
            probs={"benign": 95.0, "pre_cancerous": 3.0, "malignant": 2.0},
            image_jpg_bytes=b"",
            timestamp="2026-08-10T00:00:00",
            inference_ms=10,
            urgency="",
            icon="",
            action="",
            backend_id="mock",
        )


def _jpeg(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _field(rgb, size: int = SIZE) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = rgb
    return img


def _textured(rgb, size: int = SIZE) -> np.ndarray:
    """A surface with real detail, so it passes the focus check on its own."""
    base = _field(rgb, size).astype(np.int16)
    rng = np.random.default_rng(7)
    base += rng.integers(-28, 28, base.shape, dtype=np.int16)
    return np.clip(base, 0, 255).astype(np.uint8)


def _skin_with_lesion() -> np.ndarray:
    img = _textured(SKIN)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    m = (yy - SIZE // 2) ** 2 + (xx - SIZE // 2) ** 2 <= 55**2
    img[m] = LESION
    return img


def _run(rgb: np.ndarray, backend: _Backend | None = None):
    b = backend or _Backend()
    return run_pipeline(b, _jpeg(rgb), pixels_per_mm=10.0, strict=False), b


def test_a_real_lesion_photo_reaches_the_classifier() -> None:
    pl, backend = _run(_skin_with_lesion())
    assert backend.calls == 1
    assert not pl.get("blocked")
    assert pl["verdict"].state == "low_concern"


@pytest.mark.parametrize(
    "surface", [(240, 240, 240), (128, 128, 128), (30, 80, 200), (60, 150, 60)]
)
def test_non_skin_surfaces_never_reach_the_classifier(surface) -> None:
    """A 3-class softmax always sums to 1, so anything that gets through comes
    back as a confident label. The gate is what stops that."""
    pl, backend = _run(_textured(surface))
    assert backend.calls == 0, "junk input reached the model"
    assert pl["blocked"]
    assert pl["verdict"].state == "no_lesion"


def test_wall_is_told_it_is_not_skin_not_that_it_is_blurry() -> None:
    """Ordering matters: a featureless wall fails the focus check too, but
    "hold the camera still" is useless advice for it."""
    pl, _ = _run(_field((235, 235, 235)))
    assert pl["verdict"].state == "no_lesion"
    assert "blurry" not in pl["verdict"].text().lower()


def test_bare_skin_with_no_spot_is_stopped_with_its_own_message() -> None:
    pl, backend = _run(_textured(SKIN))
    assert backend.calls == 0
    assert pl["verdict"].state == "no_lesion"


def test_blurry_skin_is_told_to_retake_not_that_there_is_no_skin() -> None:
    smeared = cv2.GaussianBlur(_skin_with_lesion(), (31, 31), 0)
    pl, backend = _run(smeared)
    assert backend.calls == 0
    assert pl["verdict"].state == "retake"


def test_slightly_soft_lesion_photo_still_gets_a_result() -> None:
    """The complaint that started this: a usable photo must not be thrown away."""
    soft = cv2.GaussianBlur(_skin_with_lesion(), (5, 5), 0)
    pl, backend = _run(soft)
    assert backend.calls == 1, f"soft photo rejected: {pl.get('verdict')}"
    assert not pl.get("blocked")


def _dim(rgb: np.ndarray) -> np.ndarray:
    """Dim enough to earn an advisory, not dark enough to be unreadable."""
    return (rgb.astype(np.float32) * 0.16).astype(np.uint8)


def test_a_dim_photo_is_advisory_by_default() -> None:
    pl, backend = _run(_dim(_skin_with_lesion()))
    assert backend.calls == 1, f"dim photo blocked by default: {pl.get('verdict')}"
    assert any(code == "dark" for code, _l, _s in pl["quality"]["reason_details"])


def test_strict_mode_blocks_the_same_dim_photo() -> None:
    """The toggle has to actually do something, in the direction it claims."""
    b = _Backend()
    pl = run_pipeline(b, _jpeg(_dim(_skin_with_lesion())), pixels_per_mm=10.0, strict=True)
    assert b.calls == 0
    assert pl["blocked"] and pl["verdict"].state == "retake"


def test_both_backends_apply_the_same_rules() -> None:
    """The Pi path once ignored strict entirely, so the same photo
    passed on one device and was rejected on the other."""

    class _PiBackend(_Backend):
        backend_id = "pi"

        def scan(self, image_jpg_bytes=None):
            sr = super().scan(image_jpg_bytes)
            return type(sr)(**{**sr.__dict__, "image_jpg_bytes": self._jpg})

    dim = _jpeg(_dim(_skin_with_lesion()))
    pi = _PiBackend()
    pi._jpg = dim
    strict_pi = run_pipeline(pi, None, pixels_per_mm=10.0, strict=True)
    strict_upload = run_pipeline(_Backend(), dim, pixels_per_mm=10.0, strict=True)
    assert strict_pi["verdict"].state == strict_upload["verdict"].state == "retake"


def test_every_stop_state_carries_a_verdict_the_screen_can_render() -> None:
    for rgb in (_field((235, 235, 235)), _textured(SKIN), cv2.GaussianBlur(_skin_with_lesion(), (31, 31), 0)):
        pl, _ = _run(rgb)
        v = pl.get("verdict")
        assert v is not None and v.headline and v.body and v.advice
        assert pl.get("rgb") is not None, "the stop screen still shows the photo"
