"""Synthetic tests for ABCDE geometry helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from services.abcde import (
    asymmetry_score,
    border_score,
    colour_score,
    compute_abcde,
    diameter_mm,
)
from services.segmentation import segment_safe


def _circle_mask(h: int = 256, w: int = 256, r: int = 40) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (w // 2, h // 2), r, 255, -1)
    return m


def test_circle_low_asymmetry() -> None:
    m = _circle_mask()
    a = asymmetry_score(m)
    assert a < 0.12


def test_circle_compactness_near_one() -> None:
    m = _circle_mask()
    b = border_score(m)
    assert 0.95 <= b <= 1.15


def test_asymmetric_mask_higher_asymmetry() -> None:
    m = np.zeros((128, 128), dtype=np.uint8)
    m[20:100, 20:55] = 255
    m[20:100, 73:108] = 255
    a_sym = asymmetry_score(m)
    circ = _circle_mask(128, 128, 35)
    a_circ = asymmetry_score(circ)
    assert a_sym >= a_circ


def test_diameter_mm_scales() -> None:
    m = _circle_mask(r=50)
    d10 = diameter_mm(m, pixels_per_mm=10.0)
    d5 = diameter_mm(m, pixels_per_mm=5.0)
    assert d5 > d10 > 0


def test_compute_abcde_uniform_color() -> None:
    h, w = 128, 128
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = (180, 120, 100)
    m = _circle_mask(h, w, 30)
    out = compute_abcde(rgb, m, pixels_per_mm=10.0)
    assert out["C"]["tier"] == 0
    assert out["E"]["verdict"] == "needs history"


def test_segment_safe_fallback_mask() -> None:
    rgb = np.full((200, 200, 3), 200, dtype=np.uint8)
    m = segment_safe(rgb)
    assert m is not None
    assert np.count_nonzero(m) > 0
