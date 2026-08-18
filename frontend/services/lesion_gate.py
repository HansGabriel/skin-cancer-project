"""Decide whether a photo actually shows a skin spot worth screening.

This is a *content* gate, deliberately separate from ``services.quality`` (a
*capture* gate). They answer different questions and deserve different messages:

- quality  → "I can see skin, but this photo is hard to read." → take another
- this file → "There is no skin spot here at all."             → point at a spot

Without it, a photo of a wall or a keyboard still reaches the classifier, and a
3-class softmax always sums to 1 — so the model confidently reports "benign".

Two stages run on every capture, both cheap enough for a Raspberry Pi 4:

1. **Is this skin?** YCrCb chroma bounds do the work. Separating brightness (Y)
   from colour (Cr/Cb) is what makes one test work across light and deeply
   pigmented skin. The previous HSV-only test required ``V >= 60`` and
   ``S <= 180`` — a band tuned for light skin that rejected darker Filipino skin
   tones outright, which is why real lesions came back as "not enough skin".
2. **Is there a spot on it?** Reuses the existing segmentation mask, then checks
   the blob is a plausible size, is not the entire frame, and is actually
   *different* in colour from the skin around it. Uniform skin with no lesion
   gives a near-zero difference.

An optional third stage compares the model's own features against statistics
built from the training set (``models/feature_stats.json``, produced by
``scripts/build_feature_stats.py``). It is skipped when that file is absent, so
the gate degrades to stages 1–2 rather than failing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# Fraction of the frame that must look like skin.
#
# Measured on 200 real HAM10000 images (2026-08-13): the old 0.12 rejected 11%
# of genuine lesions, but lowering this number was NOT the fix — the Cb ceiling
# in skin_mask was (see there). With Cb at 140 the real rejection rate is 2.0%
# anywhere from 0.06 to 0.12, so the threshold is chosen for junk protection
# instead: 0.08 sits well above the worst neutral surface in
# tests/test_gate_robustness.py (textured dark grey through JPEG, 0.057) and
# far below the weakest real skin tone there (0.797).
#
# Do not drop this to 0.05 "to be safe" — that is below the grey-desk score and
# buys 0.7pp of real images. The regression test will catch it.
_SKIN_MIN = float(os.environ.get("SKIN_GATE_SKIN_MIN", "0.08"))
# A lesion smaller than this is noise; larger than this is the whole frame.
_LESION_MIN_FRAC = float(os.environ.get("SKIN_GATE_LESION_MIN_FRAC", "0.004"))
_LESION_MAX_FRAC = float(os.environ.get("SKIN_GATE_LESION_MAX_FRAC", "0.75"))
# Coverage a mask must ALSO reach before "touches every edge" counts as
# "the spot fills the whole photo" — see _touches_all_borders.
_LESION_BORDER_FRAC = float(os.environ.get("SKIN_GATE_LESION_BORDER_FRAC", "0.50"))
# Lab colour distance between the spot and the skin ring around it. ~2.3 is the
# classic "just noticeable difference"; below this there is no distinct spot.
_MIN_CONTRAST = float(os.environ.get("SKIN_GATE_MIN_CONTRAST", "5.0"))
# Mahalanobis distance beyond which the model's features look nothing like the
# training data. Unset by default: the threshold is read from the stats file's
# own p99 instead, because a bare numeric default is a trap — 0 would reject
# every image, and any fixed number is meaningless against a covariance that
# depends on which images the stats were built from.
_MAX_OOD_ENV = os.environ.get("SKIN_GATE_MAX_OOD")

_WORK_PX = 384  # analysis resolution — keeps the gate fast on the Pi


@dataclass(frozen=True)
class FrameCheck:
    """What the gate saw. ``reasons`` are already in plain language."""

    is_lesion_photo: bool
    skin_fraction: float
    lesion_fraction: float
    lesion_contrast: float
    reasons: tuple[str, ...] = ()

    @property
    def has_skin(self) -> bool:
        return self.skin_fraction >= _SKIN_MIN


def _work_size(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if max(h, w) <= _WORK_PX:
        return image_rgb
    scale = _WORK_PX / float(max(h, w))
    return cv2.resize(image_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _normalise_exposure(image_rgb: np.ndarray, target: float = 110.0) -> np.ndarray:
    """Lift a dim frame to a standard brightness before judging its colour.

    Skin is identified by *chroma*, and a JPEG taken in dim light has its chroma
    crushed toward neutral — so a real lesion shot in a poorly lit hall came
    back as "no skin was found", the same class of false rejection this gate
    exists to remove. Brightening first makes the test exposure-independent,
    which is what lets the saturation floor stay high enough to keep rejecting
    grey surfaces. Bright frames are left alone (gain is never below 1).
    """
    v = float(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)[:, :, 2].mean())
    if v <= 1.0 or v >= target:
        return image_rgb
    gain = min(target / v, 4.0)
    return cv2.convertScaleAbs(image_rgb, alpha=gain, beta=0)


def skin_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that look like human skin, at any melanin level.

    YCrCb carries the tone-independent part of the test; the widened HSV bounds
    only exclude greys and blues (walls, screens, sky, denim) that happen to
    fall inside the chroma box.
    """
    image_rgb = _normalise_exposure(image_rgb)
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.int16)
    cb = ycrcb[:, :, 2].astype(np.int16)
    # Cb ceiling is 140, not the textbook 127. Polarized and oil-immersion
    # dermoscopy renders skin pink-violet under a cool white balance and lands
    # at Cb 133-138 — just outside the classic box — so the images this device
    # is *for* were the ones being called "not skin". Widening to 140 took the
    # false rejection rate on real HAM10000 images from 10% to 2% without any
    # measured junk surface (wall, sky, leaf, denim, grey desk) crossing over.
    # Known cost: light-pink surfaces (e.g. pink plastic) now read as skin. They
    # are stopped a stage later by check_spot instead, which needs a distinct
    # spot with real Lab contrast — a uniform sheet has none.
    m_chroma = (cr >= 136) & (cr <= 177) & (cb >= 77) & (cb <= 140)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    # V floor of 20 (not 60) is the fix for dark skin under kiosk lighting.
    # The saturation floor of 30 is the fix for the opposite error: JPEG chroma
    # quantisation nudges near-neutral pixels toward the skin box, so a grey
    # desk photographed and saved as JPEG read as 15% skin — over the threshold
    # — while its raw pixels read as 5%. Measured through the real capture path,
    # not on raw arrays, which is the only way that shows up.
    m_hue = ((h <= 27) | (h >= 158)) & (s >= 30) & (s <= 210) & (v >= 20)
    return m_chroma & m_hue


def skin_fraction(image_rgb: np.ndarray) -> float:
    """Share of the frame that looks like skin, in [0, 1]."""
    small = _work_size(image_rgb)
    m = skin_mask(small)
    return float(np.count_nonzero(m)) / float(m.size)


def has_skin(image_rgb: np.ndarray) -> bool:
    """Stage 1 on its own, so callers can ask it before the quality checks.

    Order matters for the message the user gets: a photo of a wall is both
    "not skin" and often "unreadable", and only the first is useful to hear.
    """
    return skin_fraction(image_rgb) >= _SKIN_MIN


def _lab(image_rgb: np.ndarray) -> np.ndarray:
    """True-ish CIE Lab: L in [0,100], a/b centred on 0."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def lesion_contrast(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Lab colour distance between the segmented spot and the skin ringing it.

    A mole differs from the skin around it. Flat skin, a fingertip, or a blank
    wall does not — which is what tells "no spot here" apart from "a spot the
    model is unsure about".
    """
    m = mask > 0
    if not m.any():
        return 0.0
    k = max(3, int(0.06 * max(image_rgb.shape[:2])) | 1)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1) > 0
    ring &= ~m
    if not ring.any():
        return 0.0
    lab = _lab(image_rgb)
    inside = lab[m].mean(axis=0)
    outside = lab[ring].mean(axis=0)
    return float(np.linalg.norm(inside - outside))


def _touches_all_borders(mask: np.ndarray) -> bool:
    """A blob bleeding off every edge *and* covering most of the frame is the
    whole scene, not a lesion.

    The area half of that sentence used to be missing, and the edge test alone
    is far weaker than it sounds: a lesion that merely reaches all four edges
    through thin extensions was refused as "fills the whole photo" at only ~30%
    coverage — measured on 4 of 60 real HAM10000 images. It also contradicted
    ``segmentation._FRAC_MAX``, which happily produces masks up to 0.96.
    Requiring real coverage as well keeps the "camera pointed at an arm"
    rejection while letting large, irregular lesions through.
    """
    m = mask > 0
    if float(np.count_nonzero(m)) / float(m.size) < _LESION_BORDER_FRAC:
        return False
    return bool(m[0, :].any() and m[-1, :].any() and m[:, 0].any() and m[:, -1].any())


@lru_cache(maxsize=1)
def _feature_stats() -> dict | None:
    """Training-set feature statistics for the optional third stage."""
    path = os.environ.get("SKIN_FEATURE_STATS")
    if not path:
        root = Path(__file__).resolve().parents[2]
        path = str(root / "models" / "feature_stats.json")
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def feature_distance(features: np.ndarray) -> float | None:
    """Mahalanobis distance from the training-set feature centroid.

    Returns ``None`` when no statistics have been built, so callers treat the
    stage as absent rather than as a pass.
    """
    stats = _feature_stats()
    if not stats:
        return None
    try:
        mean = np.asarray(stats["mean"], dtype=np.float64)
        inv_cov = np.asarray(stats["inv_cov"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    v = np.asarray(features, dtype=np.float64).ravel()
    if v.shape != mean.shape:
        return None
    d = v - mean
    return float(np.sqrt(max(0.0, d @ inv_cov @ d)))


def _ood_threshold() -> float | None:
    """Distance beyond which a photo is treated as unlike the training data."""
    if _MAX_OOD_ENV:
        try:
            return float(_MAX_OOD_ENV)
        except ValueError:
            return None
    stats = _feature_stats() or {}
    p99 = (stats.get("train_distance_percentiles") or {}).get("99")
    return float(p99) * 1.1 if p99 else None


def ood_stage_available() -> bool:
    """True when stage 3 can actually run (``models/feature_stats.json`` exists).

    Callers use this to decide whether the conv features are worth computing at
    all. That matters because those features cost a second full forward pass:
    without this predicate, "skip the CAM to save time" would silently disable a
    safety gate, and with it the cost is paid only when the gate is real.
    """
    return _ood_threshold() is not None


def is_out_of_distribution(features: np.ndarray) -> bool | None:
    """Stage 3: does this look like anything the model was trained on?

    ``None`` means the stage did not run — no ``models/feature_stats.json``
    (build it with ``scripts/build_feature_stats.py``). Callers must treat that
    as "not checked" rather than as a pass, so the gate degrades to stages 1-2
    instead of silently approving everything.
    """
    threshold = _ood_threshold()
    if threshold is None:
        return None
    distance = feature_distance(features)
    if distance is None:
        return None
    return distance > threshold


def check_skin(image_rgb: np.ndarray) -> tuple[FrameCheck | None, float]:
    """Stage 1 alone: ``(failure_or_None, skin_fraction)``.

    Split out so the pipeline can ask this *before* the quality checks — a
    photo of a wall is both "not skin" and "unreadable", and only the first is
    useful to be told. Returns the fraction so stage 2 does not measure it a
    second time on every scan.
    """
    small = _work_size(image_rgb)
    skin = float(np.count_nonzero(skin_mask(small))) / float(small.shape[0] * small.shape[1])
    if skin >= _SKIN_MIN:
        return None, skin
    return FrameCheck(False, skin, 0.0, 0.0, ("No skin was found in this photo.",)), skin


def check_spot(image_rgb: np.ndarray, mask: np.ndarray | None, *, skin: float = 1.0) -> FrameCheck:
    """Stage 2 alone: is there a distinct spot on skin already known to be there?

    ``skin`` is passed in by callers that already measured it (the pipeline),
    so the mask and colour work is not repeated once per scan.
    """
    reasons: list[str] = []
    if mask is None or not (mask > 0).any():
        reasons.append("No spot could be picked out on the skin.")
        return FrameCheck(False, skin, 0.0, 0.0, tuple(reasons))

    frac = float(np.count_nonzero(mask > 0)) / float(mask.size)
    contrast = lesion_contrast(image_rgb, mask)

    if frac < _LESION_MIN_FRAC:
        reasons.append("The spot is too small in the frame. Move the camera closer.")
    elif frac > _LESION_MAX_FRAC or _touches_all_borders(mask):
        reasons.append("The spot fills the whole photo. Move the camera back a little.")
    elif contrast < _MIN_CONTRAST:
        reasons.append("This looks like plain skin with no clear spot on it.")

    return FrameCheck(not reasons, skin, frac, contrast, tuple(reasons))


# Lab-L spread (99.5th minus 0.5th percentile, after a median blur) below which
# a frame holds no distinct spot at all. Measured on synthetic frames matching
# tests/test_pipeline_gate.py: bare skin scores 6.7-7.5 across light, dark and
# pale tones; any lesion at or above the gate's own minimum size scores 52.5;
# the faintest lesion tested (a low-contrast tan blob) scores 18.0. 12 sits well
# clear of bare skin and well below the faintest real spot.
#
# The median blur is what makes it work on small lesions: it removes per-pixel
# capture noise, which otherwise dominates the percentiles, while leaving a blob
# of any size intact.
_PRECHECK_MIN_SPREAD = float(os.environ.get("SKIN_GATE_PRECHECK_SPREAD", "12.0"))


def _tone_spread(image_rgb: np.ndarray) -> float:
    """How much lighter-to-darker range the frame actually contains.

    Exposure-normalised first, for the same reason ``skin_mask`` does it: a dim
    capture has its whole tonal range compressed, so a real lesion shot in poor
    indoor light scores like bare skin and would be refused as "no clear spot".
    """
    lab_l = _lab(_normalise_exposure(image_rgb))[:, :, 0]
    smooth = cv2.medianBlur(lab_l, 5).astype(np.float32)
    return float(np.percentile(smooth, 99.5) - np.percentile(smooth, 0.5))


def quick_reject(image_rgb: np.ndarray, *, skin: float = 1.0) -> FrameCheck | None:
    """Cheap "is there obviously no lesion here?" test, or None to keep going.

    This exists for one measured problem. Stage 2 (:func:`check_spot`) needs a
    mask, and the mask used to be produced by the full segmentation stack —
    colour constancy, hair removal, then GrabCut at native resolution. For a
    photo *with a lesion* that is work worth doing. For a photo of a bare
    forearm it is not, and it is exactly the case that pays the most: a
    non-lesion frame produces degenerate threshold candidates, which puts the
    segmentation score out of band, which is the one condition that makes
    GrabCut always run. Measured at 4032x3024: enhance 3.7 s + segment 103 s,
    all of it spent to reach a refusal.

    Two refusals are available here, and both are chosen so that being wrong is
    not possible in the direction that matters:

    * **"plain skin, no clear spot"** — decided from :func:`_tone_spread`, which
      needs no mask at all. That is the point: the obvious implementation asks
      ``segment_safe`` for a mask, but that helper substitutes a *centre circle*
      when no candidate is plausible, and measuring contrast on that circle
      compares skin against skin for any off-centre lesion. It would refuse
      precisely the frames the expensive path exists to rescue.
    * **"fills the whole photo"** — geometry, which downscaling does not change.
      Only trusted when the cheap mask is in band; a degenerate mask means no
      opinion.

    **"too small in the frame" is never used here.** A small lesion is what the
    enhancement and GrabCut exist to find, so the coarse pass must not reject
    one. (A speck below ``_LESION_MIN_FRAC`` can still fall under the spread
    threshold and be refused as "plain skin" — the full path refuses it too,
    just with a different sentence.)

    Returning ``None`` means "no opinion" — the full path runs and decides.
    """
    work = _work_size(image_rgb)

    if _tone_spread(work) < _PRECHECK_MIN_SPREAD:
        return FrameCheck(
            False,
            skin,
            0.0,
            0.0,
            ("This looks like plain skin with no clear spot on it.",),
        )

    # segment(), not segment_safe(): see the docstring — the "safe" wrapper's
    # centre-circle fallback is not something to draw conclusions from.
    from services.segmentation import _FRAC_MAX, _FRAC_MIN, _foreground_fraction, segment

    mask = segment(work, use_grabcut=False)
    if mask is None or not (mask > 0).any():
        return None
    frac = _foreground_fraction(mask)
    if not (_FRAC_MIN <= frac <= _FRAC_MAX):
        return None
    if frac > _LESION_MAX_FRAC or _touches_all_borders(mask):
        return FrameCheck(
            False,
            skin,
            frac,
            0.0,
            ("The spot fills the whole photo. Move the camera back a little.",),
        )
    return None


def check_frame(image_rgb: np.ndarray, mask: np.ndarray | None) -> FrameCheck:
    """Both content stages in order. ``mask`` comes from ``services.segmentation``."""
    no_skin, skin = check_skin(image_rgb)
    if no_skin is not None:
        return no_skin
    return check_spot(image_rgb, mask, skin=skin)
