"""A B C on the gate's outline: a round one-colour spot reads normal, a ragged two-colour one does not."""

from __future__ import annotations

import cv2
import numpy as np

from conftest import SKIN_TONES, textured, through_jpeg, with_lesion
from dermascan import gate, signs


def _measure(rgb: np.ndarray):
    small = gate.shrink(rgb)
    report = gate.check(rgb)
    return signs.measure(small, report.outline)


def test_a_round_even_spot_reads_normal() -> None:
    out = {s.letter: s for s in _measure(through_jpeg(with_lesion(textured(SKIN_TONES["light"]))))}
    assert out["A"].tier == 0 and out["B"].tier == 0 and out["C"].tier == 0
    assert out["D"].tier is None and out["E"].tier is None and out["E"].value is None


def test_a_ragged_two_colour_spot_stands_out() -> None:
    img = textured(SKIN_TONES["light"])
    pts = np.array([[150, 140], [330, 120], [300, 200], [360, 330], [220, 300], [130, 360], [170, 240]], np.int32)
    cv2.fillPoly(img, [pts], (70, 50, 60))
    cv2.circle(img, (290, 160), 28, (150, 90, 60), -1)  # a second, browner colour inside
    out = {s.letter: s for s in _measure(through_jpeg(img))}
    assert out["A"].tier >= 1 and out["B"].tier >= 1 and out["C"].tier >= 1


def test_no_outline_means_no_signs() -> None:
    assert signs.measure(np.zeros((10, 10, 3), np.uint8), None) is None
    assert signs.measure(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool)) is None


def test_same_photo_same_colour_count() -> None:
    rgb = through_jpeg(with_lesion(textured(SKIN_TONES["medium"])))
    assert [s.to_dict() for s in _measure(rgb)] == [s.to_dict() for s in _measure(rgb)]
