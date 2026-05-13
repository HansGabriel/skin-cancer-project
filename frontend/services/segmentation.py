"""Lesion mask: multi-candidate LAB/HSV Otsu + morphology (OpenCV only).

Dermoscopic images often have very dark lesions; a single inverted-L Otsu path
frequently misses or over-segments. We try several grayscale views and pick
the mask whose foreground fraction is closest to a plausible lesion size.
"""

from __future__ import annotations

import cv2
import numpy as np

# Foreground fraction must fall in this band to accept (dermoscopy-friendly).
_FRAC_MIN = 0.015
_FRAC_MAX = 0.96
_IDEAL_FRAC = 0.18


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


def _grabcut_center_mask(rgb: np.ndarray) -> np.ndarray:
    """GrabCut with a central ROI — fallback when global thresholds fail."""
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return np.zeros((h, w), dtype=np.uint8)
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


def segment(image_rgb: np.ndarray) -> np.ndarray:
    """Return binary uint8 mask (0/255) using the best-scoring grayscale candidate."""
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
        _grabcut_center_mask(image_rgb),
    )

    best: np.ndarray | None = None
    best_score = 1e9
    for m in candidates:
        frac = _foreground_fraction(m)
        s = _score_mask(frac)
        if s < best_score:
            best_score = s
            best = m

    if best is None:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    return best


def segment_safe(image_rgb: np.ndarray) -> np.ndarray | None:
    """Return mask only if foreground fraction is plausible; else ``None``."""
    mask = segment(image_rgb)
    frac = _foreground_fraction(mask)
    if frac < _FRAC_MIN or frac > _FRAC_MAX:
        return None
    return mask
