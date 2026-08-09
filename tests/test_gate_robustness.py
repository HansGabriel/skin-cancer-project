"""Robustness of the content gate to how a photo was actually captured.

Every bug these cover was invisible to raw-array tests and only appeared once
the image went through the real encode, or was taken in poor light — which is
exactly the situation this kiosk runs in.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from services.lesion_gate import has_skin, is_out_of_distribution, skin_fraction

SIZE = 320
SKIN_TONES = {"light": (241, 194, 165), "medium": (198, 134, 66), "deep": (61, 42, 33)}
NEUTRAL = {"grey desk": (120, 120, 125), "white wall": (240, 240, 240), "dark grey": (70, 70, 72)}
# Below this the frame is essentially black (mean brightness ~22) and the light
# meter already reports it; the gate is not expected to recover colour there.
EXPOSURES = (1.0, 0.5, 0.3, 0.22, 0.16)


def _textured(rgb: tuple[int, int, int], amp: int = 28) -> np.ndarray:
    rng = np.random.default_rng(7)
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:, :] = rgb
    img = img.astype(np.int16) + rng.integers(-amp, amp, (SIZE, SIZE, 3), dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _dim(rgb: np.ndarray, factor: float) -> np.ndarray:
    return (rgb.astype(np.float32) * factor).astype(np.uint8)


def _through_jpeg(rgb: np.ndarray, quality: int = 92) -> np.ndarray:
    """The exact encode the upload path performs before anything sees the image."""
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    with Image.open(io.BytesIO(png.tobytes())) as im:
        rgb_im = im.convert("RGB")
    buf = io.BytesIO()
    rgb_im.save(buf, format="JPEG", quality=quality)
    decoded = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("factor", EXPOSURES)
@pytest.mark.parametrize("tone,rgb", SKIN_TONES.items(), ids=list(SKIN_TONES))
def test_skin_is_found_at_any_exposure(tone: str, rgb, factor: float) -> None:
    """A real lesion photographed in a dim hall must not be told "no skin".

    JPEG crushes chroma toward neutral in low light, so without normalising
    exposure first the skin test failed on genuinely valid photos — the same
    class of false rejection the gate was built to remove.
    """
    img = _through_jpeg(_dim(_textured(rgb), factor))
    assert has_skin(img), f"{tone} at {factor:.0%} light: {skin_fraction(img):.3f}"


@pytest.mark.parametrize("factor", EXPOSURES)
@pytest.mark.parametrize("name,rgb", NEUTRAL.items(), ids=list(NEUTRAL))
def test_neutral_surfaces_stay_rejected_at_any_exposure(name: str, rgb, factor: float) -> None:
    """Brightening dim frames must not turn a grey desk into skin."""
    img = _through_jpeg(_dim(_textured(rgb), factor))
    assert not has_skin(img), f"{name} at {factor:.0%} light: {skin_fraction(img):.3f}"


def test_out_of_distribution_stage_is_absent_not_passing_without_stats(monkeypatch) -> None:
    """No statistics file means "not checked" — never a silent approval."""
    from services import lesion_gate

    monkeypatch.setenv("SKIN_FEATURE_STATS", "/nonexistent/feature_stats.json")
    lesion_gate._feature_stats.cache_clear()
    assert is_out_of_distribution(np.zeros(8)) is None
    lesion_gate._feature_stats.cache_clear()


def test_out_of_distribution_flags_features_far_from_training(monkeypatch, tmp_path) -> None:
    """With statistics present, familiar features pass and foreign ones do not."""
    import json

    from services import lesion_gate

    stats = {
        "mean": [0.0] * 4,
        "inv_cov": np.eye(4).tolist(),
        "train_distance_percentiles": {"99": 3.0},
    }
    path = tmp_path / "feature_stats.json"
    path.write_text(json.dumps(stats))
    monkeypatch.setenv("SKIN_FEATURE_STATS", str(path))
    monkeypatch.delenv("SKIN_GATE_MAX_OOD", raising=False)
    lesion_gate._feature_stats.cache_clear()

    assert is_out_of_distribution(np.zeros(4)) is False  # dead centre
    assert is_out_of_distribution(np.array([50.0, 0, 0, 0])) is True  # far outside
    lesion_gate._feature_stats.cache_clear()


def test_results_photo_is_discarded_on_every_route_change(monkeypatch) -> None:
    """docs/PRIVACY.md promises this, and it regressed once already: "Done"
    cleared the photo but the bottom navigation called navigate() directly, so
    tapping Home from a result left the previous participant's skin behind."""
    import navigation

    state: dict = {}
    monkeypatch.setattr(navigation.st, "session_state", state)

    for destination in ("home", "camera", "history", "settings", "folder"):
        state.clear()
        state.update({"route": "results", "last_result": {"rgb": "a photo"}})
        navigation._discard_result_on_leave(destination)
        assert "last_result" not in state, f"photo survived navigating to {destination}"


def test_screens_that_work_with_the_scan_keep_it(monkeypatch) -> None:
    """The assistant answers questions about the result, and the case screen is
    where saving lands — both would break if the photo were dropped."""
    import navigation

    state: dict = {}
    monkeypatch.setattr(navigation.st, "session_state", state)

    for destination in ("assistant", "case", "results"):
        state.clear()
        state.update({"route": "results", "last_result": {"rgb": "a photo"}})
        navigation._discard_result_on_leave(destination)
        assert "last_result" in state, f"{destination} needs the scan and lost it"
