"""ABCDE heuristic scores (A–D from image + mask; E needs history — stub)."""

from __future__ import annotations

from typing import Any, TypedDict

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Tier thresholds (Notion Part 4.3)
TIER_A = (0.15, 0.30)  # normal < lo, border lo–hi, suspicious > hi
TIER_B = (1.8, 2.5)
TIER_D_MM = (4.0, 6.0)


class LetterResult(TypedDict):
    value: float | int | None
    tier: int
    verdict: str


def _tier_from_float(x: float, lo: float, hi: float) -> int:
    if x < lo:
        return 0
    if x <= hi:
        return 1
    return 2


def _tier_from_mm(mm: float) -> int:
    if mm < TIER_D_MM[0]:
        return 0
    if mm <= TIER_D_MM[1]:
        return 1
    return 2


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def orient_mask_crop(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Crop to lesion bbox and rotate so major axis is horizontal (for asymmetry)."""
    cnt = _largest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < 4:
        return mask, None
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    if angle < -45:
        angle += 90
    h, w = mask.shape[:2]
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return mask, None
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.where(rot > 0)
    if len(xs) == 0:
        return rot, None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rot[y0:y1, x0:x1]
    return crop, cnt


def asymmetry_score(mask: np.ndarray) -> float:
    """0 symmetric, 1 highly asymmetric (mask-only heuristic)."""
    crop, _ = orient_mask_crop(mask)
    if crop.size == 0 or np.count_nonzero(crop) == 0:
        return 0.0
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return 0.0
    m = (crop > 0).astype(np.uint8)
    area = max(int(m.sum()), 1)
    h2 = h // 2
    top, bot = m[:h2, :], np.flipud(m[h - h2 :, :])
    rw = min(top.shape[1], bot.shape[1])
    top, bot = top[:, :rw], bot[:, :rw]
    v_asym = float(np.sum(top != bot)) / float(area)
    w2 = w // 2
    left, right = m[:, :w2], np.fliplr(m[:, w - w2 :])
    rh = min(left.shape[0], right.shape[0])
    left, right = left[:rh, :], right[:rh, :]
    h_asym = float(np.sum(left != right)) / float(area)
    return float(min(1.0, (v_asym + h_asym) / 2.0))


def border_score(mask: np.ndarray) -> float:
    """Compactness P^2/(4*pi*A); circle ~= 1, irregular > 1."""
    cnt = _largest_contour(mask)
    if cnt is None:
        return 1.0
    peri = cv2.arcLength(cnt, closed=True)
    area = cv2.contourArea(cnt)
    if area < 1e-6:
        return 1.0
    return float((peri**2) / (4.0 * np.pi * area + 1e-6))


def _delta_e_lab(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def colour_distinct_count(image_rgb: np.ndarray, mask: np.ndarray, k: int = 5) -> int:
    """Count LAB clusters separated by ΔE > threshold (15)."""
    sel = image_rgb[mask > 0]
    if len(sel) < k:
        k = max(1, len(sel))
    if len(sel) == 0:
        return 0
    lab_px = cv2.cvtColor(sel.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
    km.fit(lab_px)
    centers = km.cluster_centers_
    used = [False] * len(centers)
    groups = 0
    for i in range(len(centers)):
        if used[i]:
            continue
        groups += 1
        for j in range(i + 1, len(centers)):
            if not used[j] and _delta_e_lab(centers[i], centers[j]) <= 15.0:
                used[j] = True
    return groups


def colour_score(image_rgb: np.ndarray, mask: np.ndarray, k: int = 5) -> int:
    return colour_distinct_count(image_rgb, mask, k=k)


def diameter_mm(mask: np.ndarray, pixels_per_mm: float) -> float:
    """Equivalent diameter from largest contour area."""
    cnt = _largest_contour(mask)
    if cnt is None or pixels_per_mm <= 0:
        return 0.0
    area_px = cv2.contourArea(cnt)
    if area_px <= 0:
        return 0.0
    d_px = 2.0 * np.sqrt(area_px / np.pi)
    return float(d_px / pixels_per_mm)


def compute_abcde(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    pixels_per_mm: float = 10.0,
) -> dict[str, LetterResult]:
    a_raw = asymmetry_score(mask)
    b_raw = border_score(mask)
    c_raw = colour_score(image_rgb, mask)
    d_mm = diameter_mm(mask, pixels_per_mm)

    c_tier = 0 if c_raw <= 1 else (1 if c_raw == 2 else 2)

    return {
        "A": {
            "value": round(a_raw, 4),
            "tier": _tier_from_float(a_raw, TIER_A[0], TIER_A[1]),
            "verdict": ("normal" if a_raw < TIER_A[0] else "borderline" if a_raw <= TIER_A[1] else "suspicious"),
        },
        "B": {
            "value": round(b_raw, 3),
            "tier": _tier_from_float(b_raw, TIER_B[0], TIER_B[1]),
            "verdict": ("normal" if b_raw < TIER_B[0] else "borderline" if b_raw <= TIER_B[1] else "suspicious"),
        },
        "C": {
            "value": int(c_raw),
            "tier": c_tier,
            "verdict": ("normal" if c_raw <= 1 else "borderline" if c_raw == 2 else "suspicious"),
        },
        "D": {
            "value": round(d_mm, 2),
            "tier": _tier_from_mm(d_mm),
            "verdict": ("normal" if d_mm < TIER_D_MM[0] else "borderline" if d_mm <= TIER_D_MM[1] else "suspicious"),
        },
        "E": {
            "value": None,
            "tier": 0,
            "verdict": "needs history",
        },
    }


def abcde_tier_sum_ad(abcde: dict[str, LetterResult]) -> int:
    """Sum tiers for A–D (for composite risk /8)."""
    return int(sum(abcde[k]["tier"] for k in ("A", "B", "C", "D")))
