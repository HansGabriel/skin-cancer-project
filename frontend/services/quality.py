"""Capture-quality checks — "can this photo be read?", in plain language.

Two levels, on purpose. Most imperfect photos are still perfectly readable, so
they produce an **advisory** the user sees next to a real result. Only a photo
that is genuinely unusable is a **hard** stop.

That split is the fix for a specific complaint: a real lesion, shot slightly
soft or in warm indoor light, used to be rejected outright with "image not
clear" and no result at all. Now it is scanned, and the note appears alongside.

Whether the photo shows *skin* at all is not asked here — that is
``services.lesion_gate``. Mixing "hard to read" with "not a skin spot" is what
made every failure produce the same unhelpful message.
"""

from __future__ import annotations

import os
from typing import TypedDict

import cv2
import numpy as np

# Focus thresholds, measured on REAL dermoscopy (2026-08-13: 200 random
# HAM10000 images) — see docs/METRICS.md "Capture quality thresholds".
#
#   real HAM10000   median 79, p25 55, p10 42
#   samples/*.jpg   449 / 435 / 469
#
# The previous soft default of 120 was tuned against samples/, which
# samples/README.md documents as "synthetic placeholders (simple color
# patches)". Synthetic patches score ~5x real dermoscopy, so 120 sat far above
# the median real photo and **72.5% of real lesions tripped the advisory** —
# and because an advisory blocks whenever strict mode is on, most real lesions
# were refused outright. That is the same mistake the old default of 35 made
# (tuned on a checkerboard), one order of magnitude worse.
#
# 40 puts the advisory at ~9.5% of real dermoscopy: soft enough to still flag a
# genuinely bad photo, low enough that ordinary lesions get scanned. The hard
# stop is 20 rather than 25 because a real HAM10000 image measured 24.8 — a
# photo that loses by 0.2 is not "unusable".
_BLUR_SOFT = float(os.environ.get("SKIN_QUALITY_BLUR_MIN", "40"))
_BLUR_HARD = float(os.environ.get("SKIN_QUALITY_BLUR_HARD", "20"))

_V_SOFT_MIN = float(os.environ.get("SKIN_QUALITY_V_MIN", "35"))
_V_SOFT_MAX = float(os.environ.get("SKIN_QUALITY_V_MAX", "220"))
_V_HARD_MIN = float(os.environ.get("SKIN_QUALITY_V_HARD_MIN", "12"))
_V_HARD_MAX = float(os.environ.get("SKIN_QUALITY_V_HARD_MAX", "246"))

# Focus is measured at one bounded width so the number means the same thing for
# a 4056px Pi capture and a 640px webcam frame. Downscale only — upscaling a
# small image interpolates detail away and makes a sharp photo look blurry.
_FOCUS_PX = 512


class QualityResult(TypedDict):
    ok: bool  # False only for a hard stop
    reasons: list[str]  # plain-language advisories and hard failures
    reason_details: list[tuple[str, str, str]]  # (code, chip label, severity)


def _focus_score(image_rgb: np.ndarray) -> float:
    """Sharpness, independent of both capture resolution and exposure.

    Two normalisations, each fixing a way the raw Laplacian variance lies:

    - **Size.** Downscale to at most 512px. Variance grows with resolution, so
      without this the same scene grades differently from the Pi camera and a
      webcam. Downscale only — upscaling a small capture interpolates its
      detail away and makes a sharp photo look soft.
    - **Contrast.** Stretch to the full range first. Variance scales with the
      square of contrast, so a dim-but-sharp photo would otherwise be rejected
      as "blurry" when the real problem is the light — which ``light_meter``
      already reports separately, and which is only advisory.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > _FOCUS_PX:
        scale = _FOCUS_PX / float(max(h, w))
        gray = cv2.resize(
            gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    if int(gray.max()) - int(gray.min()) > 4:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())


def _brightness(image_rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 2].mean())


def check_quality(image_rgb: np.ndarray) -> QualityResult:
    """Grade one capture. ``ok=False`` means the photo cannot be used at all."""
    reasons: list[str] = []
    details: list[tuple[str, str, str]] = []
    hard = False

    focus = _focus_score(image_rgb)
    if focus < _BLUR_HARD:
        reasons.append("This photo is too blurry to read. Hold the camera still and try again.")
        details.append(("blur", "Out of focus", "hard"))
        hard = True
    elif focus < _BLUR_SOFT:
        reasons.append("This photo is a little blurry, so the result is less reliable.")
        details.append(("blur", "Slightly blurry", "warning"))

    v = _brightness(image_rgb)
    if v < _V_HARD_MIN:
        reasons.append("This photo is too dark to read. Move into brighter light.")
        details.append(("dark", "Too dark", "hard"))
        hard = True
    elif v > _V_HARD_MAX:
        reasons.append("This photo is washed out by light. Move out of the glare.")
        details.append(("bright", "Too bright", "hard"))
        hard = True
    elif v < _V_SOFT_MIN:
        reasons.append("The light is dim, so the result is less reliable.")
        details.append(("dark", "Dim light", "warning"))
    elif v > _V_SOFT_MAX:
        reasons.append("The light is very bright, so the result is less reliable.")
        details.append(("bright", "Bright light", "warning"))

    return {"ok": not hard, "reasons": reasons, "reason_details": details}


def focus_meter(image_rgb: np.ndarray) -> tuple[int, str]:
    """0–3 focus bars plus a one-word label, for live feedback before capture."""
    focus = _focus_score(image_rgb)
    if focus < _BLUR_HARD:
        return 0, "blurry"
    if focus < _BLUR_SOFT:
        return 1, "soft"
    if focus < _BLUR_SOFT * 2:
        return 2, "okay"
    return 3, "sharp"


def focus_score(image_rgb: np.ndarray) -> float:
    """Public sharpness reading, for calibration scripts and diagnostics."""
    return _focus_score(image_rgb)


def light_meter(image_rgb: np.ndarray) -> tuple[int, str]:
    """0–3 light bars plus a one-word label, for live feedback before capture."""
    v = _brightness(image_rgb)
    if v < _V_HARD_MIN:
        return 0, "too dark"
    if v > _V_HARD_MAX:
        return 0, "too bright"
    if v < _V_SOFT_MIN:
        return 1, "dim"
    if v > _V_SOFT_MAX:
        return 1, "harsh"
    if 60 <= v <= 190:
        return 3, "good"
    return 2, "okay"
