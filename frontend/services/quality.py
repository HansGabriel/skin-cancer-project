"""Capture quality checks with structured reason codes."""

from __future__ import annotations

import os
from typing import TypedDict

import cv2
import numpy as np

_BLUR_MIN = float(os.environ.get("SKIN_QUALITY_BLUR_MIN", "35"))
_V_MIN = float(os.environ.get("SKIN_QUALITY_V_MIN", "35"))
_V_MAX = float(os.environ.get("SKIN_QUALITY_V_MAX", "220"))
_SKIN_MIN = float(os.environ.get("SKIN_QUALITY_SKIN_MIN", "0.15"))


class QualityResult(TypedDict):
    ok: bool
    reasons: list[str]
    reason_details: list[tuple[str, str, str]]


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _skin_fraction_hsv(image_rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([179, 180, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    skin = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(skin > 0)) / float(skin.size)


def check_quality(image_rgb: np.ndarray) -> QualityResult:
    reasons: list[str] = []
    details: list[tuple[str, str, str]] = []
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if _laplacian_var(gray) < _BLUR_MIN:
        msg = "Image too blurry — please refocus and try again."
        reasons.append(msg)
        details.append(("blur", "🔍 Out of focus", "warning"))
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v_mean = float(hsv[:, :, 2].mean())
    if v_mean < _V_MIN:
        msg = "Lighting too dark — adjust illumination."
        reasons.append(msg)
        details.append(("dark", "🔆 Too dark", "warning"))
    elif v_mean > _V_MAX:
        msg = "Lighting too bright — adjust illumination."
        reasons.append(msg)
        details.append(("bright", "🔆 Too bright", "warning"))
    if _skin_fraction_hsv(image_rgb) < _SKIN_MIN:
        msg = "Not enough skin in frame — center the lesion on skin."
        reasons.append(msg)
        details.append(("skin_frac", "📏 Lesion framing", "info"))
    return {"ok": len(reasons) == 0, "reasons": reasons, "reason_details": details}
