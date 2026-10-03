"""The gate: skin, readable, one spot. All fixtures go through a real JPEG encode."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import NEUTRAL, SKIN_TONES, textured, through_jpeg, with_lesion
from dermascan import gate

EXPOSURES = (1.0, 0.5, 0.3, 0.22, 0.16)


def _dim(rgb: np.ndarray, factor: float) -> np.ndarray:
    return (rgb.astype(np.float32) * factor).astype(np.uint8)


@pytest.mark.parametrize("factor", EXPOSURES)
@pytest.mark.parametrize("tone,rgb", SKIN_TONES.items(), ids=list(SKIN_TONES))
def test_skin_is_found_at_any_exposure(tone, rgb, factor) -> None:
    """A real lesion photographed in a dim hall must not be told "no skin"."""
    img = through_jpeg(_dim(textured(rgb), factor))
    assert gate.skin_fraction(img) >= gate.config.SKIN_MIN_FRACTION, f"{tone} at {factor:.0%}"


@pytest.mark.parametrize("factor", EXPOSURES)
@pytest.mark.parametrize("name,rgb", NEUTRAL.items(), ids=list(NEUTRAL))
def test_neutral_surfaces_are_not_skin_at_any_exposure(name, rgb, factor) -> None:
    img = through_jpeg(_dim(textured(rgb), factor))
    assert gate.skin_fraction(img) < gate.config.SKIN_MIN_FRACTION, f"{name} at {factor:.0%}"


@pytest.mark.parametrize("tone,rgb", SKIN_TONES.items(), ids=list(SKIN_TONES))
def test_a_lesion_on_any_skin_tone_passes(tone, rgb) -> None:
    colour = (78, 62, 74) if tone != "deep" else (230, 200, 190)  # pale mark on deep skin
    report = gate.check(through_jpeg(with_lesion(textured(rgb), colour)))
    assert report.passed, f"{tone}: {report.refusal}"
    assert gate.config.SPOT_MIN_FRACTION <= report.measured["spot_fraction"] <= gate.config.SPOT_MAX_FRACTION


def test_bare_skin_is_refused_softly() -> None:
    report = gate.check(through_jpeg(textured(SKIN_TONES["medium"])))
    assert not report.passed
    assert report.refusal.code in ("no_spot", "plain_skin", "fills_frame")
    assert report.refusal.can_override


def test_bare_skin_under_uneven_light_is_still_refused() -> None:
    img = textured(SKIN_TONES["light"], size=600)
    ramp = np.linspace(0.6, 1.15, img.shape[1], dtype=np.float32)[None, :, None]
    img = np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    report = gate.check(through_jpeg(img))
    assert not report.passed, report.measured


def test_bare_skin_under_a_soft_shadow_is_refused_as_a_shadow() -> None:
    img = textured(SKIN_TONES["light"], size=600)
    m = np.zeros((600, 600), np.float32)
    cv2.circle(m, (280, 320), 200, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 60)
    img = np.clip(img.astype(np.float32) * (1 - 0.35 * m[:, :, None]), 0, 255).astype(np.uint8)
    report = gate.check(through_jpeg(img))
    assert report.refusal is not None and report.refusal.code == "soft_edge", report.measured
    assert report.refusal.can_override


@pytest.mark.parametrize("sigma", (0, 3, 6, 10))
def test_a_softly_focused_lesion_still_has_a_real_edge(sigma) -> None:
    """Measured on the edge check alone: a synthetic frame blurred this much also
    trips the focus floor, which real skin (with texture) does not."""
    img = with_lesion(textured(SKIN_TONES["light"], size=600))
    if sigma:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    small = gate.shrink(through_jpeg(img))
    width = gate.edge_width(small, gate.outline_spot(small))
    # Clear of the threshold by 2x up to a 6 px blur; a 10 px blur must still pass.
    limit = gate.config.SPOT_MAX_EDGE_WIDTH_PCT / (2 if sigma <= 6 else 1)
    assert 0 < width < limit, width


def test_a_wall_is_refused_hard() -> None:
    report = gate.check(through_jpeg(textured(NEUTRAL["white wall"])))
    assert report.refusal.code == "no_skin"
    assert not report.refusal.can_override


def test_a_blurred_photo_is_refused_before_the_spot_check() -> None:
    img = with_lesion(textured(SKIN_TONES["light"]))
    img = cv2.GaussianBlur(img, (0, 0), 14)
    report = gate.check(through_jpeg(img))
    assert report.refusal.code == "too_blurry"
    assert "spot_fraction" not in report.measured


def test_a_black_frame_is_too_dark() -> None:
    img = with_lesion(_dim(textured(SKIN_TONES["light"]), 0.04))
    assert gate.check(through_jpeg(img)).refusal.code in ("too_dark", "no_skin")


def test_a_spot_filling_the_frame_is_too_close() -> None:
    img = with_lesion(textured(SKIN_TONES["light"]), radius_frac=0.7)
    report = gate.check(through_jpeg(img))
    assert report.refusal.code == "fills_frame"
    assert report.refusal.can_override


def test_the_gate_is_fast_at_any_resolution() -> None:
    import time

    img = with_lesion(textured(SKIN_TONES["light"], size=2000))
    t = time.perf_counter()
    gate.check(img)
    assert time.perf_counter() - t < 0.5  # x86 budget; the Pi is ~10x slower and still under 5 s total


# --- Failures measured on 267 real HAM10000 lesions (2026-10-03), rebuilt synthetically:
# the repo ships no dataset, so each fixture copies what the real photo did to the outline.


def _hairy(img: np.ndarray, n: int = 40, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(n):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(out, p1, p2, (40, 28, 26), 2)
    return out


def test_a_mole_under_dense_hair_is_outlined_not_the_hairs() -> None:
    """Hairs crossing a mole used to become the outline: a thin web refused as "plain skin"."""
    img = _hairy(with_lesion(textured(SKIN_TONES["light"], size=600), colour=(150, 90, 80)))
    report = gate.check(through_jpeg(img))
    assert report.passed, (report.refusal, report.measured)
    assert report.measured["spot_solidity"] >= 0.8


def test_a_pale_mole_beside_a_dark_frame_edge_is_still_found() -> None:
    """A dark vignette at the edge used to win Otsu's threshold, and the brown mole was ignored."""
    img = with_lesion(textured((205, 170, 160), size=600), colour=(160, 110, 80), radius_frac=0.1)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.clip((xx - 0.82 * w) / (0.12 * w), 0, 1)[..., None]  # darkens towards the right edge
    img = (img.astype(np.float32) * (1 - 0.8 * edge)).astype(np.uint8)
    report = gate.check(through_jpeg(img))
    assert report.passed, (report.refusal, report.measured)
    assert not gate._touches_border(report.outline)


def test_a_real_mole_edge_is_not_called_a_shadow() -> None:
    """Real lesions measure edge width up to ~4.9; shadows 6-13. The limit sits between."""
    assert 4.9 < gate.config.SPOT_MAX_EDGE_WIDTH_PCT < 6.0
