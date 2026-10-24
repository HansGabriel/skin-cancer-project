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


def test_one_glare_dot_does_not_make_a_sharp_photo_blurry() -> None:
    """The old min/max stretch let a single specular highlight cut the focus score by a third."""
    img = with_lesion(textured(SKIN_TONES["light"], size=600))
    glared = img.copy()
    cv2.circle(glared, (180, 180), 12, (255, 255, 255), -1)
    a, b = gate.focus_score(through_jpeg(img)), gate.focus_score(through_jpeg(glared))
    assert b >= 0.85 * a, (a, b)
    assert gate.check(through_jpeg(glared)).refusal is None


def test_heavy_blur_is_still_refused() -> None:
    img = cv2.GaussianBlur(with_lesion(textured(SKIN_TONES["light"], size=600)), (0, 0), 8)
    assert gate.check(through_jpeg(img)).refusal.code == "too_blurry"


# --- One spot, in the ring (2026-10-20): a webcam face and a palm passed every check
# above and the model called both cancer. Rebuilt synthetically, as the old app's
# tests/test_not_a_lesion.py (commit 246c072) did: no faces of real people in the repo.


def _skin_field(size: int = 600) -> np.ndarray:
    return textured(SKIN_TONES["light"], size=size)


def test_a_spot_running_off_the_frame_edge_is_not_in_the_ring() -> None:
    img = _skin_field()
    cv2.circle(img, (552, 300), 90, (78, 62, 74), -1)
    report = gate.check(through_jpeg(img))
    assert report.refusal is not None and report.refusal.code == "not_in_ring", report.measured
    assert report.refusal.can_override
    assert report.measured["spot_at_edge"] == 1.0


def test_a_spot_in_the_corner_is_not_in_the_ring() -> None:
    img = _skin_field()
    cv2.circle(img, (90, 90), 48, (78, 62, 74), -1)
    report = gate.check(through_jpeg(img))
    assert report.refusal is not None and report.refusal.code == "not_in_ring", report.measured
    assert report.measured["spot_offset"] > gate.config.SPOT_MAX_OFFSET


def test_a_spot_a_little_off_centre_still_passes() -> None:
    """The ring the visitor sees has a radius of ~0.40 of the side; 0.20 is well inside it."""
    img = _skin_field()
    cv2.circle(img, (300 + int(0.2 * 600), 300), 60, (78, 62, 74), -1)
    report = gate.check(through_jpeg(img))
    assert report.passed, (report.refusal, report.measured)


def _face_like(size: int = 600) -> np.ndarray:
    """Skin with two eyes, two nostrils and a mouth: several dark marks sharing the dark area."""
    img = textured((196, 158, 132), size=size)
    cv2.ellipse(img, (200, 220), (45, 18), 0, 0, 360, (60, 45, 50), -1)
    cv2.ellipse(img, (400, 220), (45, 18), 0, 0, 360, (60, 45, 50), -1)
    cv2.circle(img, (275, 330), 16, (80, 60, 60), -1)
    cv2.circle(img, (325, 330), 16, (80, 60, 60), -1)
    cv2.ellipse(img, (300, 430), (70, 12), 0, 0, 360, (120, 70, 70), -1)
    return img


def test_a_face_is_several_marks_not_a_spot() -> None:
    report = gate.check(through_jpeg(_face_like()))
    assert report.refusal is not None and report.refusal.code == "several_marks", report.measured
    assert report.refusal.can_override
    assert report.measured["spot_dominance"] < gate.config.SPOT_MIN_DOMINANCE


def test_a_mole_with_one_satellite_is_still_one_spot() -> None:
    """Dominance, not a raw count: one mole holding most of the dark area passes."""
    img = with_lesion(_skin_field())
    cv2.circle(img, (430, 200), 28, (78, 62, 74), -1)
    report = gate.check(through_jpeg(img))
    assert report.passed, (report.refusal, report.measured)
    assert report.measured["other_marks"] == 1.0 and report.measured["spot_dominance"] > 0.9


def test_a_half_shadowed_hand_is_refused() -> None:
    """The palm: its shadowed half was the "spot", 0.46 of the frame, under the old 0.75 limit.
    Here the right half darkens over a soft ramp, the way a hand curls away from the light."""
    img = _skin_field()
    ramp = np.clip((np.arange(600) - 300) / 60.0, 0, 1)
    img = np.clip(img.astype(np.float32) * (1 - 0.35 * ramp[None, :, None]), 0, 255).astype(np.uint8)
    report = gate.check(through_jpeg(img))
    assert not report.passed, report.measured
    assert report.refusal.code in ("fills_frame", "not_in_ring", "soft_edge")


def test_a_dermoscopy_sized_lesion_still_fits_the_frame() -> None:
    """Dermoscopy uploads cover 0.3-0.6 of the frame; the 0.40 ceiling must keep the smaller ones."""
    report = gate.check(through_jpeg(with_lesion(_skin_field(), radius_frac=0.33)))
    assert report.passed, (report.refusal, report.measured)
    assert 0.3 < report.measured["spot_fraction"] < gate.config.SPOT_MAX_FRACTION


def test_every_passing_photo_reports_where_the_spot_sits() -> None:
    report = gate.check(through_jpeg(with_lesion(_skin_field())))
    assert report.passed
    for key in ("spot_offset", "spot_at_edge", "other_marks", "spot_dominance", "ring_skin"):
        assert isinstance(report.measured[key], float), key
    assert set(gate.measure_everything(through_jpeg(_face_like()))) >= set(report.measured)
