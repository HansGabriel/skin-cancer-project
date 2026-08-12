"""Capture quality has two levels, and the split is the point.

An imperfect-but-readable photo must still produce a result with a note beside
it. Rejecting those outright is what made real lesions come back as "image not
clear" and nothing else.

Note what ``samples/`` actually is: ``samples/README.md`` documents those three
files as **synthetic placeholders (simple colour patches)**. They score 435-469
on the normalised focus metric, while real HAM10000 dermoscopy has a median of
79. Calibrating against them is how the soft threshold reached 120 and refused
72% of genuine lesions — the same mistake, one order of magnitude worse, as the
checkerboard-tuned 35 that preceded it.

So the tests here are a floor, not a calibration: they prove the two levels
behave sanely on shapes we control. The thresholds themselves are calibrated
against real dermoscopy, and ``tests/test_gate_real_images.py`` is what holds
that line.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from services.quality import check_quality, focus_meter, light_meter

SAMPLES = sorted((Path(__file__).resolve().parent.parent / "samples").glob("*.jpg"))


def _load(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _checkerboard(size: int, squares: int = 16, value: int = 200) -> np.ndarray:
    step = max(1, size // squares)
    yy, xx = np.mgrid[:size, :size]
    board = (((yy // step) + (xx // step)) % 2).astype(np.uint8) * value + 20
    return cv2.cvtColor(board, cv2.COLOR_GRAY2RGB)


def _flat(size: int, value: int) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


@pytest.mark.skipif(not SAMPLES, reason="no sample images in samples/")
@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_sample_images_are_never_rejected(path: Path) -> None:
    """Sanity floor on the shipped demo images.

    Named for what it does: these are the synthetic ``samples/`` patches, not
    real lesions. The real-photo guarantee lives in test_gate_real_images.py.
    """
    q = check_quality(_load(path))
    assert q["ok"], f"{path.name} rejected: {q['reasons']}"
    assert focus_meter(_load(path))[0] >= 2, f"{path.name} graded as soft/blurry"


@pytest.mark.skipif(not SAMPLES, reason="no sample images in samples/")
@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_a_genuinely_blurred_lesion_photo_is_caught(path: Path) -> None:
    """The gate still has to work — the same photo, actually blurred, fails."""
    smeared = cv2.GaussianBlur(_load(path), (11, 11), 0)
    assert not check_quality(smeared)["ok"]
    assert focus_meter(smeared)[0] == 0


def test_sharp_well_lit_photo_passes_clean() -> None:
    q = check_quality(_checkerboard(512))
    assert q["ok"] and not q["reasons"]


def test_featureless_image_is_a_hard_stop() -> None:
    q = check_quality(_flat(512, 128))  # no detail at all
    assert not q["ok"]
    assert any(sev == "hard" for _c, _l, sev in q["reason_details"])


def test_pitch_dark_is_a_hard_stop() -> None:
    q = check_quality(_flat(512, 4))
    assert not q["ok"]
    assert any(code == "dark" and sev == "hard" for code, _l, sev in q["reason_details"])


@pytest.mark.skipif(not SAMPLES, reason="no sample images in samples/")
def test_dim_but_visible_light_is_only_advisory() -> None:
    """A real lesion shot in dim indoor light still gets scanned, with a note."""
    dim = (_load(SAMPLES[0]).astype(np.float32) * 0.14).astype(np.uint8)
    q = check_quality(dim)
    assert q["ok"], f"dim photo blocked: {q['reasons']}"
    assert any(code == "dark" and sev == "warning" for code, _l, sev in q["reason_details"])


def test_focus_score_is_independent_of_capture_resolution() -> None:
    """The same scene from the 4056px Pi camera and a 640px webcam must grade
    the same. Laplacian variance scales with resolution, so the un-normalised
    version was effectively a different test per capture path."""
    assert focus_meter(_checkerboard(1024)) == focus_meter(_checkerboard(2048))


def test_small_images_are_never_upscaled_into_looking_blurry() -> None:
    """Upscaling a 224px capture to 512 interpolates its detail away."""
    small = _checkerboard(224, squares=8)
    assert focus_meter(small)[0] == 3


def test_meters_span_their_range() -> None:
    assert focus_meter(_checkerboard(512))[0] == 3
    assert focus_meter(_flat(512, 128))[0] == 0
    assert light_meter(_flat(512, 120))[0] == 3
    assert light_meter(_flat(512, 4))[0] == 0
    assert light_meter(_flat(512, 252))[0] == 0


def test_messages_avoid_jargon() -> None:
    jargon = ("laplacian", "variance", "hsv", "threshold", "px", "channel")
    for img in (_flat(512, 4), _flat(512, 128)):
        for reason in check_quality(img)["reasons"]:
            assert not any(j in reason.lower() for j in jargon), reason
            assert reason[0].isupper() and reason.endswith(".")
