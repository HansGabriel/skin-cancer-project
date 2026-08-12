"""GrabCut is skipped when it cannot help — and still runs when it can.

GrabCut used to be computed on every scan as the 7th mask candidate. Measured
at the 1024x1024 frame the Pi actually captures, the six threshold candidates
cost ~78 ms together and GrabCut alone cost ~2200 ms: 99% of segmentation, and
roughly 85% of a 30-second scan on a Pi 4.

What makes skipping safe is a property of the scoring, not a hope about images:
``_score_mask`` depends **only** on foreground fraction, and ``segment`` breaks
ties toward the earlier (cheap) candidate. So GrabCut can only change the
outcome by scoring strictly lower, and refusing to run it while the best cheap
score is already within ``_GRABCUT_SKIP_SCORE`` of perfect bounds what we can
give up at exactly that margin. These tests pin that reasoning down, because it
is the whole justification for the change.

Measured on 60 real HAM10000 images at the shipped margin: GrabCut skipped on
23%, and in the two cases where it would have won, the mask actually chosen
scored 0.0021 worse — a fifth of a percentage point of frame area.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from services.segmentation import (
    _GRABCUT_SKIP_SCORE,
    _adaptive_mask,
    _foreground_fraction,
    _grabcut_center_mask,
    _mask_from_gray,
    _score_mask,
    segment,
)


def _cheap_candidates(rgb: np.ndarray) -> list[np.ndarray]:
    """The six candidates ``segment`` builds before considering GrabCut."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v_ch = hsv[:, :, 2]
    return [
        _mask_from_gray(255 - l_ch),
        _mask_from_gray(l_ch),
        _mask_from_gray(255 - v_ch),
        _mask_from_gray(v_ch),
        _adaptive_mask(255 - v_ch),
        _adaptive_mask(255 - l_ch),
    ]


def _best_cheap_score(rgb: np.ndarray) -> float:
    return min(_score_mask(_foreground_fraction(m)) for m in _cheap_candidates(rgb))


def _lesion_frame(size: int = 256, radius: int = 60) -> np.ndarray:
    """Tan skin with a dark, clearly-bounded spot — the easy case."""
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), (215, 175, 150), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (70, 45, 40), -1)
    return img


def test_score_depends_only_on_foreground_fraction() -> None:
    """The property the whole skip argument rests on.

    Two masks of completely different shape but equal area must score the same.
    If this ever stops holding, the bound in ``segment``'s docstring is void and
    the skip has to be re-justified from scratch.
    """
    a = np.zeros((200, 200), np.uint8)
    cv2.circle(a, (100, 100), 50, 255, -1)
    # Same pixel count, deliberately nothing like the same shape: a scattered
    # stripe rather than a disc. Built by exact count so the areas are equal,
    # not merely close.
    b = np.zeros(200 * 200, np.uint8)
    b[: np.count_nonzero(a)] = 255
    b = b.reshape(200, 200)

    assert _foreground_fraction(a) == _foreground_fraction(b)
    assert _score_mask(_foreground_fraction(a)) == _score_mask(_foreground_fraction(b))


def test_grabcut_is_skipped_when_a_cheap_mask_is_already_good() -> None:
    """The fast path: a clean lesion must not pay for GrabCut."""
    rgb = _lesion_frame()
    assert _best_cheap_score(rgb) <= _GRABCUT_SKIP_SCORE, (
        "this fixture is meant to be an easy case for the cheap candidates; "
        "if it is not, the test below proves nothing"
    )

    calls: list[int] = []
    real = _grabcut_center_mask

    import services.segmentation as seg

    def _spy(image_rgb):  # pragma: no cover - should never run
        calls.append(1)
        return real(image_rgb)

    seg._grabcut_center_mask = _spy
    try:
        segment(rgb)
    finally:
        seg._grabcut_center_mask = real

    assert not calls, "GrabCut ran even though a cheap candidate was already good"


def test_grabcut_still_runs_when_every_cheap_candidate_fails() -> None:
    """The slow path must survive: this is the case GrabCut exists for.

    A frame where global thresholding has nothing to latch onto drives every
    cheap candidate out of band (score ~10.18), and there the fallback has to
    run — skipping it would be the actual regression.
    """
    rng = np.random.default_rng(3)
    rgb = rng.integers(90, 165, (256, 256, 3), dtype=np.uint8)
    if _best_cheap_score(rgb) <= _GRABCUT_SKIP_SCORE:
        pytest.skip("fixture did not defeat the cheap candidates")

    calls: list[int] = []
    real = _grabcut_center_mask

    import services.segmentation as seg

    def _spy(image_rgb):
        calls.append(1)
        return real(image_rgb)

    seg._grabcut_center_mask = _spy
    try:
        segment(rgb)
    finally:
        seg._grabcut_center_mask = real

    assert calls, "GrabCut was skipped on a frame the cheap candidates could not segment"


def test_skipping_costs_no_more_than_the_declared_margin() -> None:
    """The bound itself, asserted rather than argued.

    Whatever ``segment`` returns, its score is within ``_GRABCUT_SKIP_SCORE`` of
    what the full seven-candidate search would have chosen.
    """
    for fixture in (_lesion_frame(), _lesion_frame(radius=30), _lesion_frame(radius=95)):
        chosen = _score_mask(_foreground_fraction(segment(fixture)))
        exhaustive = min(
            _best_cheap_score(fixture),
            _score_mask(_foreground_fraction(_grabcut_center_mask(fixture))),
        )
        assert chosen - exhaustive <= _GRABCUT_SKIP_SCORE + 1e-9, (
            f"skip cost {chosen - exhaustive:.4f}, above the declared bound "
            f"{_GRABCUT_SKIP_SCORE}"
        )
