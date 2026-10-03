"""The A B C D E warning signs, measured on the spot the gate already outlined.

Doctors look at five things on a mole. This file measures the three it can from
one photo, on the outline gate.check drew (no second segmentation, a few ms):

    A  asymmetry   fold the outline along its long and short axes; the share
                   that does not line up (0 = mirror image)
    B  border      perimeter^2 / (4 pi area); a perfect circle is 1.0, a ragged
                   edge is higher
    C  colour      how many clearly different colour groups are inside the spot
                   (k-means in CIE Lab, groups closer than 15 merged)
    D  diameter    NOT measured: millimetres need a scale this camera does not
                   have. The share of the frame the spot covers is kept for staff.
    E  evolving    NOT measured: needs an earlier photo of the same spot.

These numbers never change the verdict. The model and thresholds.json decide;
the signs help a person see what to look at. The definitions and tiers are the
old app's (see config.py), so the paper can compare like with like.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dermascan import config


@dataclass(frozen=True)
class Sign:
    letter: str
    value: float | None  # None when it cannot be measured from one photo
    tier: int | None  # 0 normal, 1 borderline, 2 stands out, None not measured

    def to_dict(self) -> dict:
        v = None if self.value is None else round(self.value, 3)
        return {"letter": self.letter, "value": v, "tier": self.tier}


def _tier(x: float, tiers: tuple[float, float]) -> int:
    lo, hi = tiers
    return 0 if x < lo else 1 if x <= hi else 2


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def asymmetry(mask: np.ndarray) -> float:
    """Turn the outline so its long axis is level, fold it both ways, average the misfit."""
    cnt = _largest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < 4:  # a few pixels: nothing to fold
        return 0.0
    (cx, cy), _, angle = cv2.minAreaRect(cnt)
    h, w = mask.shape
    turn = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    level = cv2.warpAffine(mask.astype(np.uint8), turn, (w, h), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(level)
    if len(xs) == 0:
        return 0.0
    m = level[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    area = max(int(m.sum()), 1)
    up_down = np.count_nonzero(m != np.flipud(m)) / 2
    left_right = np.count_nonzero(m != np.fliplr(m)) / 2
    return float(min(1.0, (up_down + left_right) / 2 / area))


def border(mask: np.ndarray) -> float:
    """Perimeter squared over 4 pi area: 1.0 for a circle, higher for a ragged edge."""
    cnt = _largest_contour(mask)
    if cnt is None:
        return 1.0
    area = cv2.contourArea(cnt)
    if area < 1e-6:
        return 1.0
    return float(cv2.arcLength(cnt, True) ** 2 / (4.0 * np.pi * area))


def colour_groups(rgb_small: np.ndarray, mask: np.ndarray) -> int:
    """How many clearly different colours are inside the spot.

    Counted INSIDE: the rim where spot and skin blend (and JPEG smears them)
    would otherwise always add a second, in-between colour. A spot too small to
    lose its rim is counted whole.
    """
    k_px = 2 * config.COLOUR_RIM_PX + 1
    inner = cv2.erode(mask.astype(np.uint8), np.ones((k_px, k_px), np.uint8)).astype(bool)
    pixels = rgb_small[inner if inner.sum() >= config.COLOUR_GROUPS_K * 10 else mask]
    if len(pixels) == 0:
        return 0
    k = min(config.COLOUR_GROUPS_K, len(pixels))
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100.0 / 255.0  # OpenCV's 8-bit Lab -> real Lab units, so 15 means 15
    lab[:, 1:] -= 128.0
    cv2.setRNGSeed(42)  # same photo, same answer
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)  # k-means stops: 20 rounds or settled
    _, _, centres = cv2.kmeans(lab, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    groups: list[np.ndarray] = []
    for c in centres:
        if all(np.linalg.norm(c - g) > config.COLOUR_DISTINCT_DE for g in groups):
            groups.append(c)
    return len(groups)


def measure(rgb_small: np.ndarray, mask: np.ndarray | None) -> list[Sign] | None:
    """The five signs, or None when there is no outline worth measuring.

    `rgb_small` must be the same size as `mask` (the gate's GATE_WORK_PX copy).
    """
    if mask is None or not mask.any():
        return None
    a = asymmetry(mask)
    b = border(mask)
    c = colour_groups(rgb_small, mask)
    return [
        Sign("A", a, _tier(a, config.ASYMMETRY_TIERS)),
        Sign("B", b, _tier(b, config.BORDER_TIERS)),
        Sign("C", float(c), _tier(c, config.COLOUR_TIERS)),
        Sign("D", float(mask.mean()), None),  # share of the frame, not millimetres
        Sign("E", None, None),
    ]
