"""Lesion mask: multi-candidate LAB/HSV Otsu + morphology (OpenCV only).

Dermoscopic images often have very dark lesions; a single inverted-L Otsu path
frequently misses or over-segments. We try several grayscale views and pick
the mask whose foreground fraction is closest to a plausible lesion size.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# Foreground fraction must fall in this band to accept (dermoscopy-friendly).
_FRAC_MIN = 0.015
_FRAC_MAX = 0.96
_IDEAL_FRAC = 0.18

# How good a cheap candidate has to be before GrabCut is skipped entirely.
# Because _score_mask depends only on foreground fraction, this is a hard bound
# on what the skip can cost, not an estimate — see segment().
#
# Chosen by measurement, 20 real HAM10000 images, comparing the mask and every
# ABCD tier against the old always-run-GrabCut behaviour:
#
#   margin   GrabCut skipped   mask differs   ABCD tier flips
#   0.02          30%               0                0
#   0.05          40%               0                0
#   0.10          60%               0                0     <- shipped
#   0.20          90%               1                1
#
# 0.10 is the largest margin with no measured change of any kind; 0.20 is where
# the first tier flip appears, so that is the ceiling, not a tuning knob.
# Lower it (0 disables the skip) to always pay ~2.2 s of GrabCut per scan.
_GRABCUT_SKIP_SCORE = float(os.environ.get("SKIN_GRABCUT_SKIP_SCORE", "0.10"))


def _adaptive_mask(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold — helps uneven illumination in dermoscopy."""
    g = cv2.GaussianBlur(gray, (9, 9), 0)
    h, w = gray.shape[:2]
    bsz = max(15, min(h, w) // 12)
    if bsz % 2 == 0:
        bsz += 1
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsz, -2
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    return _largest_component_mask(th)


def _largest_component_mask(binary: np.ndarray) -> np.ndarray:
    """Keep largest 8-connected foreground; output 0/255 uint8."""
    m = (binary > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    j = 1 + int(np.argmax(areas))
    return np.where(labels == j, 255, 0).astype(np.uint8)


def _mask_from_gray(gray: np.ndarray) -> np.ndarray:
    """Otsu + morph close/open on single-channel uint8."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return _largest_component_mask(th)


def _foreground_fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask > 0)) / float(mask.size)


def _score_mask(frac: float) -> float:
    """Lower is better; in-band masks strongly preferred."""
    if _FRAC_MIN <= frac <= _FRAC_MAX:
        return abs(frac - _IDEAL_FRAC)
    return 10.0 + abs(frac - _IDEAL_FRAC)


# GrabCut initialises its two colour models with k-means, and OpenCV's k-means
# draws from one process-wide RNG whose state advances on every call. So the
# same photo segmented twice in one session could come back with a different
# mask — measured on a lesion photographed under uneven light: three identical
# calls, foreground fractions 0.276, 0.087, 0.087.
#
# That is not a tuning problem, it is a reproducibility one. Everything
# downstream reads the mask: the ABCDE letters, the diameter shown to the user,
# the risk band, and the content gate's decision to refuse. A screening device
# that answers the same photograph differently on a re-scan cannot be defended,
# and it also makes any threshold measured against the mask meaningless.
#
# The seed value itself is arbitrary and carries no meaning — only that it is
# the same one every time.
_GRABCUT_SEED = 20260831


def _grabcut_center_mask(rgb: np.ndarray) -> np.ndarray:
    """GrabCut with a central ROI — fallback when global thresholds fail."""
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return np.zeros((h, w), dtype=np.uint8)
    cv2.setRNGSeed(_GRABCUT_SEED)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rect = (w // 10, h // 10, max(1, int(0.8 * w)), max(1, int(0.8 * h)))
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), dtype=np.uint8)
    binm = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _largest_component_mask(binm)


def segment(image_rgb: np.ndarray, *, use_grabcut: bool = True) -> np.ndarray:
    """Return binary uint8 mask (0/255) using the best-scoring grayscale candidate.

    ``use_grabcut=False`` runs the cheap candidates only. That is not a quality
    setting — it is for the pipeline's pre-check, which needs a coarse mask to
    decide whether the frame holds a lesion *at all* before paying for the real
    thing. Never use it for a mask whose measurements will be displayed: the
    skip predicate below exists precisely because GrabCut is what rescues the
    cases the cheap candidates get wrong.

    GrabCut is computed **last and only when the cheap candidates are not
    already good enough**. It used to run unconditionally and it dominated the
    entire scan: measured at 1024x1024, the six threshold candidates cost ~78 ms
    together while GrabCut alone cost ~2200 ms — 99% of segmentation, and around
    85% of a 30-second scan on a Pi 4.

    The skip is safe by construction, not by luck. ``_score_mask`` is a pure
    function of foreground fraction, and the loop below breaks ties toward the
    earlier (cheap) candidate, so GrabCut can only change the outcome when it
    scores strictly lower. Skipping while the best cheap score is already within
    ``_GRABCUT_SKIP_SCORE`` of perfect therefore bounds the worst case we can
    give up at exactly that margin — for an in-band mask the score *is*
    ``|fraction - 0.18|``, so the shipped 0.10 means "the chosen mask is at most
    10 percentage points of frame area further from the ideal lesion size than
    GrabCut's". Measured on real images the realised cost is far below that
    bound: at this margin no mask changed at all (see the constant above).

    Note what is deliberately NOT the predicate: "skip if any cheap candidate is
    in band". ``_FRAC_MAX`` is 0.96, so a mask swallowing 90% of the frame is
    technically in band while scoring 0.72 — and there GrabCut genuinely should
    get its turn. Equally, when every cheap candidate is out of band (score
    ~10.18) GrabCut always runs: that is the "fallback when global thresholds
    fail" case its own docstring describes, and the case it exists for.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3 RGB uint8")
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v_ch = hsv[:, :, 2]

    candidates = (
        _mask_from_gray(255 - l_ch),
        _mask_from_gray(l_ch),
        _mask_from_gray(255 - v_ch),
        _mask_from_gray(v_ch),
        _adaptive_mask(255 - v_ch),
        _adaptive_mask(255 - l_ch),
    )

    best: np.ndarray | None = None
    best_score = 1e9
    for m in candidates:
        s = _score_mask(_foreground_fraction(m))
        if s < best_score:
            best_score = s
            best = m

    if use_grabcut and best_score > _GRABCUT_SKIP_SCORE:
        gc = _grabcut_center_mask(image_rgb)
        if _score_mask(_foreground_fraction(gc)) < best_score:
            best = gc

    if best is None:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    return best


def _center_circle_mask(image_rgb: np.ndarray, radius_frac: float = 0.35) -> np.ndarray:
    """Circular ROI fallback when contour segmentation fails."""
    h, w = image_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * radius_frac)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def segment_or_fallback(
    image_rgb: np.ndarray, *, use_grabcut: bool = True
) -> tuple[np.ndarray, bool]:
    """``(mask, is_a_guess)`` — whether the mask was found or invented.

    When no candidate is plausible this returns a plain circle in the middle of
    the frame. That keeps ABCDE from crashing, which is why it exists, but it is
    a placeholder and not a lesion outline: every measurement taken from it —
    asymmetry, border, diameter — describes a circle someone drew, not anything
    in the photo.

    ``segment_safe`` threw the distinction away, so no caller could tell the two
    apart. The consequence was measurable: ``lesion_gate.check_spot`` has a
    branch for "No spot could be picked out on the skin", and because it was
    only ever handed a mask that existed, **that branch was unreachable through
    the pipeline**. Every frame got a plausible-looking outline whatever was in
    it, which is the same failure mode as a face being read as a lesion, one
    layer down.
    """
    mask = segment(image_rgb, use_grabcut=use_grabcut)
    frac = _foreground_fraction(mask)
    if _FRAC_MIN <= frac <= _FRAC_MAX:
        return mask, False
    return _center_circle_mask(image_rgb), True


def segment_safe(image_rgb: np.ndarray, *, use_grabcut: bool = True) -> np.ndarray | None:
    """Return mask if plausible; else center-circle fallback (ABCDE never crashes).

    Kept for callers that only want a mask to measure and have already decided
    a guess is acceptable — ``services.evolving`` re-measuring an old saved
    photo, for instance. Anything deciding whether to *show a verdict* should
    call :func:`segment_or_fallback` and look at the flag.
    """
    mask, _is_a_guess = segment_or_fallback(image_rgb, use_grabcut=use_grabcut)
    return mask
