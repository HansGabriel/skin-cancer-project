"""What a refusal is allowed to cost.

The gate's *decisions* are covered by test_pipeline_gate.py. This file covers
the other half: how much work is done before it says no.

The bug it exists for was measured, not theorised. A skin-coloured photo with
no distinct spot on it — a bare forearm, the single most likely wrong thing to
point a skin scanner at — took **107 seconds** at 12 MP before refusing. None
of it was the model: 3.7 s of colour-constancy and hair removal, then 103 s of
segmentation, because a frame with no lesion produces degenerate threshold
candidates, which is exactly the condition that forces GrabCut to run at native
resolution.

Two changes fixed it, and both are pinned here: the pipeline caps its working
resolution, and a cheap pre-check answers the obvious cases before the
expensive stages run.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.contracts import ScanResult
from services.pipeline import MAX_WORK_PX, run_pipeline

SIZE = 320
SKIN = (198, 134, 66)
LESION = (44, 28, 22)

# Stages that must not have run before a "this is not a lesion photo" refusal.
EXPENSIVE = ("enhance", "segment", "model", "abcde")


class _Backend:
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
            timestamp="2026-08-19T00:00:00",
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


def _textured(rgb, h: int = SIZE, w: int = SIZE) -> np.ndarray:
    base = np.zeros((h, w, 3), dtype=np.int16)
    base[:, :] = rgb
    base += np.random.default_rng(7).integers(-28, 28, base.shape, dtype=np.int16)
    return np.clip(base, 0, 255).astype(np.uint8)


def _skin_with_lesion(h: int = SIZE, w: int = SIZE) -> np.ndarray:
    img = _textured(SKIN, h, w)
    yy, xx = np.ogrid[:h, :w]
    r = min(h, w) // 6
    img[(yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r**2] = LESION
    return img


def _run(rgb: np.ndarray):
    b = _Backend()
    return run_pipeline(b, _jpeg(rgb), pixels_per_mm=10.0, strict=False), b


# --- The pre-check --------------------------------------------------------


def test_bare_skin_is_refused_without_paying_for_segmentation() -> None:
    """The 107-second case. Refusing is right; refusing slowly was the bug."""
    pl, backend = _run(_textured(SKIN))
    assert pl["blocked"]
    assert pl["verdict"].state == "no_lesion"
    assert backend.calls == 0
    stages = pl.get("stage_ms") or {}
    for name in EXPENSIVE:
        assert name not in stages, f"{name} ran before a non-lesion refusal: {dict(stages)}"


def test_a_real_lesion_still_pays_for_the_full_path() -> None:
    """The pre-check must be an early exit, not a replacement.

    The displayed ABCDE tiers come from the full-resolution, colour-corrected,
    GrabCut-refined mask. If the cheap pre-check ever started deciding for real
    lesions too, those numbers would quietly change.
    """
    pl, backend = _run(_skin_with_lesion())
    assert not pl.get("blocked")
    assert backend.calls == 1
    stages = pl.get("stage_ms") or {}
    assert "enhance" in stages and "segment" in stages
    assert pl.get("abcde") is not None


def test_a_wall_never_reaches_the_pre_check_either() -> None:
    """Stage 1 already answers this one, and more cheaply."""
    pl, backend = _run(_textured((128, 128, 128)))
    assert pl["blocked"] and pl["verdict"].state == "no_lesion"
    assert backend.calls == 0
    stages = pl.get("stage_ms") or {}
    assert "prespot" not in stages, "the skin stage should have answered first"


def test_pre_check_never_refuses_on_size_alone() -> None:
    """A small lesion is precisely what the expensive path exists to find, so
    the coarse mask is not allowed to reject one."""
    from services.lesion_gate import quick_reject

    img = _textured(SKIN, 512, 512)
    yy, xx = np.ogrid[:512, :512]
    img[(yy - 256) ** 2 + (xx - 256) ** 2 <= 14**2] = LESION  # ~0.2% of frame
    verdict = quick_reject(img)
    if verdict is not None:
        assert "too small" not in " ".join(verdict.reasons).lower()


# --- The resolution cap ---------------------------------------------------


def test_working_resolution_is_capped_for_big_uploads() -> None:
    """A 12 MP phone photo must cost what a device capture costs.

    services/pi_camera.py fixes its stills at a 1024 centre crop, so the Pi was
    always cheap; only uploads were not. The cap lives in the pipeline rather
    than only in the upload widget so no caller can route around it.
    """
    big = _skin_with_lesion(2400, 3200)
    pl, backend = _run(big)
    assert backend.calls == 1
    assert max(pl["rgb"].shape[:2]) == MAX_WORK_PX
    assert (pl.get("stage_ms") or {}).get("resize") is not None


def test_small_captures_are_left_alone() -> None:
    """No resample, and no re-encode, for a frame already within the cap."""
    small = _skin_with_lesion(400, 300)
    pl, _ = _run(small)
    assert pl["rgb"].shape[:2] == (400, 300)
    assert "resize" not in (pl.get("stage_ms") or {})


@pytest.mark.parametrize("shape", [(2400, 3200), (3200, 2400)])
def test_cap_preserves_aspect_ratio(shape) -> None:
    """Squashing the frame would move every ABCDE measurement that follows."""
    h, w = shape
    pl, _ = _run(_skin_with_lesion(h, w))
    out_h, out_w = pl["rgb"].shape[:2]
    assert max(out_h, out_w) == MAX_WORK_PX
    assert abs((out_w / out_h) - (w / h)) < 0.01
