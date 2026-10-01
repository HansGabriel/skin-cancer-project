"""Shared fixtures: synthetic frames pushed through the real JPEG path.

Two gate bugs in this project's history were invisible on raw arrays and only
appeared after a JPEG encode (chroma quantisation, grain). So every fixture
here is encoded exactly the way a capture or an upload is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SKIN_TONES = {"light": (241, 194, 165), "medium": (198, 134, 66), "deep": (61, 42, 33)}
NEUTRAL = {"grey desk": (120, 120, 125), "white wall": (240, 240, 240), "dark grey": (70, 70, 72)}


def textured(rgb: tuple[int, int, int], size: int = 480, amp: int = 24, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), rgb, np.int16)
    img += rng.integers(-amp, amp, (size, size, 3), dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def with_lesion(base: np.ndarray, colour=(78, 62, 74), radius_frac: float = 0.17) -> np.ndarray:
    img = base.copy()
    h, w = img.shape[:2]
    cv2.circle(img, (w // 2, h // 2), int(min(h, w) * radius_frac), colour, -1)
    return img


def jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def through_jpeg(rgb: np.ndarray, quality: int = 92) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(jpeg(rgb, quality), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def classifier():
    from dermascan.classifier import Classifier

    return Classifier()
