"""Capture quality checks (blur, exposure, skin fraction). Lesion area: use ``segmentation.segment_safe`` in pipeline."""

from __future__ import annotations

import os
from typing import TypedDict

import cv2
import numpy as np

# Defaults tuned for dermoscopic / clinical photos (JPEG + resize often lowers Laplacian var).
# Override with env: SKIN_QUALITY_BLUR_MIN, SKIN_QUALITY_V_MIN, SKIN_QUALITY_V_MAX, SKIN_QUALITY_SKIN_MIN
_BLUR_MIN = float(os.environ.get("SKIN_QUALITY_BLUR_MIN", "35"))
_V_MIN = float(os.environ.get("SKIN_QUALITY_V_MIN", "35"))
_V_MAX = float(os.environ.get("SKIN_QUALITY_V_MAX", "220"))
_SKIN_MIN = float(os.environ.get("SKIN_QUALITY_SKIN_MIN", "0.15"))


class QualityResult(TypedDict):
    ok: bool
    reasons: list[str]


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _skin_fraction_hsv(image_rgb: np.ndarray) -> float:
    """Rough skin-toned pixel fraction in HSV (heuristic ranges)."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # Two broad skin clusters in OpenCV H (0–179)
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([179, 180, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    skin = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(skin > 0)) / float(skin.size)


def check_quality(image_rgb: np.ndarray) -> QualityResult:
    """Pre-segment checks: Laplacian sharpness, mean V, skin coverage."""
    reasons: list[str] = []
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lap = _laplacian_var(gray)
    if lap < _BLUR_MIN:
        reasons.append("Image too blurry — please refocus and try again.")

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v_mean = float(hsv[:, :, 2].mean())
    if v_mean < _V_MIN or v_mean > _V_MAX:
        reasons.append("Lighting too dark or too bright — adjust illumination.")

    skin_frac = _skin_fraction_hsv(image_rgb)
    if skin_frac < _SKIN_MIN:
        reasons.append("Not enough skin-toned area in frame — center the lesion on skin.")

    return {"ok": len(reasons) == 0, "reasons": reasons}
