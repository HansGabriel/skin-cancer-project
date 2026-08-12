"""Lesion image enhancement before quality / model (no weight changes)."""

from __future__ import annotations

import cv2
import numpy as np


def shades_of_gray(rgb: np.ndarray, p: float = 6.0) -> np.ndarray:
    """Finlayson color constancy on RGB uint8.

    float32 with repeated multiplication rather than float64 with ``np.power``:
    at the 1024x1024 frame the Pi actually captures, the old form allocated a
    25 MB float64 array and evaluated a transcendental ``pow`` per element,
    costing ~118 ms — on a device where memory bandwidth is the binding
    constraint. For the integer exponent this defaults to, ``x**6`` is three
    multiplications, and float32 halves the traffic. Same result to well within
    a uint8 rounding step; non-integer ``p`` still falls back to ``np.power``.
    """
    img = rgb.astype(np.float32) + 1.0
    if float(p).is_integer() and 0 < int(p) <= 8:
        powered = img.copy()
        for _ in range(int(p) - 1):
            powered *= img
    else:
        powered = np.power(img, p)
    scale = np.power(np.mean(powered, axis=(0, 1)), 1.0 / p)
    out = img / scale * 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_hair(rgb: np.ndarray) -> np.ndarray:
    """Black-hat hair mask + Telea inpaint on luminance."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def enhance_lesion_image(rgb: np.ndarray, *, dehair: bool = True) -> np.ndarray:
    out = shades_of_gray(rgb)
    if dehair:
        out = remove_hair(out)
    return out
