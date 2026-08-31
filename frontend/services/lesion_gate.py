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

   Three further checks live inside that stage, and they exist because it was
   not enough on its own: a photograph of a bare forearm under uneven light
   passed every test above and was given a verdict. They ask whether the
   outlined thing is on the person (:func:`mask_on_skin`), whether its edge
   stops the way pigment does or fades the way a shadow does
   (:func:`edge_width`), and whether it stands out from the variation this
   skin already has (:func:`contrast_z`). See the table beside
   ``_MIN_ON_SKIN``.

An optional third stage compares the model's own features against statistics
built from the training set (``models/feature_stats.json``, produced by
``scripts/build_feature_stats.py``). It is skipped when that file is absent, so
the gate degrades to stages 1–2 rather than failing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal
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
# How far past the furthest *training* image a photo has to sit before it is
# called out-of-distribution.
#
# The threshold used to be p99 x 1.1, which is the statistically tidy answer and
# the wrong one here. The statistics are built from HAM10000 — contact
# dermatoscope, polarised or oil-immersion, a 15-20mm field. The kiosk is a bare
# fixed-focus camera module under room light. Its captures are genuinely further
# from that centroid than a held-out dermoscopy image is, so a line drawn just
# past p99 would refuse real spots on day one.
#
# So the line is drawn past the *furthest image the model was actually trained
# on* (the 100th percentile), with a margin on top. That only catches gross
# outliers — a face, a wall, a screen — which is what this stage is for; the
# cheap checks above handle the rest. Raise or lower it with SKIN_GATE_MAX_OOD
# once real captures have been logged; every scan records its distance.
_OOD_MARGIN = float(os.environ.get("SKIN_GATE_OOD_MARGIN", "1.25"))

_WORK_PX = 384  # analysis resolution — keeps the gate fast on the Pi

# --- "one spot, or a scene?" ------------------------------------------------
# A lesion is one dark region on skin. A face is several — eyes, nostrils, lips
# — and that is the whole reason a face passes every other check here: stage 2
# asks "is there a distinct dark blob on this skin", and a face answers yes
# emphatically. Measured on a real portrait the segmenter landed on 0.176 of the
# frame against an ideal lesion size of 0.18, and a Lab contrast of 63 against a
# threshold of 5. It thought it had found a textbook lesion.
#
# So the discriminator is not size or contrast, it is *how the dark area is
# distributed*: one blob holding essentially all of it, or several sharing it.
# Dominance (largest blob's share of dark-blob area) is used rather than a raw
# count because it is robust to threshold noise — a single lesion scores 1.00
# whether or not speckle adds blobs around it.
#
# Measured, synthetic fixtures plus a real portrait:
#
#   single lesion            1 blob   dominance 1.00
#   multi-lobed melanoma     1 blob             1.00
#   lesion + one satellite   2 blobs            0.92
#   lesion under heavy hair  1 blob             1.00
#   lesion on dark skin      1 blob             1.00
#   five blobs on skin       3 blobs            0.58   <- refused
#   real face                10 blobs           0.50   <- refused
#   real face, tight crop    5 blobs            0.57   <- refused
#
# Both conditions must hold, and the contrast floor is the third guard: without
# it a frame with no real structure at all — bare skin, or a pale lesion barely
# distinguishable from skin — thresholds into a scatter of noise blobs with low
# dominance and would be refused. Measured there: bare skin 0.4, a pale lesion
# 2.2, every genuine spot above 27. The floor of 8 is comfortably between them.
_STRUCTURE_MIN_BLOBS = int(os.environ.get("SKIN_GATE_STRUCTURE_BLOBS", "3"))
_STRUCTURE_MAX_DOMINANCE = float(os.environ.get("SKIN_GATE_STRUCTURE_DOMINANCE", "0.65"))
_STRUCTURE_MIN_CONTRAST = float(os.environ.get("SKIN_GATE_STRUCTURE_CONTRAST", "8.0"))
# Blobs outside this size band are speckle or the background, not features.
_STRUCTURE_MIN_BLOB_FRAC = 0.004
_STRUCTURE_MAX_BLOB_FRAC = 0.60

# --- "a screen, or skin?" ---------------------------------------------------
# A display or a printed halftone carries a regular grid; skin does not. The
# grid shows up as one sharp peak in the frequency domain, far above the smooth
# falloff a photograph produces.
#
# This runs at native resolution on purpose. Measured first on the 384px working
# copy, where a screen grid scored 18 — indistinguishable from skin — because
# the resize had already thrown the grid away. Moire lives in exactly the high
# frequencies a downscale removes.
#
# Measured, worst case per class:
#
#   lesion + ruler in frame           17
#   single lesion                     18
#   multi-lobed melanoma              34
#   lesion under heavy hair           46
#   real face                         50
#   real cat / coffee photo           67
#   real face, tight crop            101
#   lesion + dermatoscope reticle    136   <- worst photograph of anything
#   ---------------------------------------
#   grid pitch 12px                  270
#   grid pitch 10px                  309
#   grid pitch  8px                  358
#   grid pitch  6px                  458
#   grid pitch  5px                  570
#   grid pitch  4px                  759
#   grid pitch  3px                  851
#   photograph OF a screen           931
#
# 400 sits 2.9x above the worst photograph and below every screen pitch a camera
# actually resolves. That margin is what makes a HARD refusal defensible here:
# the case worth fearing is a dermatoscope reticle or a ruler being called a
# screen, and the reticle — the closest any photograph came — is still a third
# of the line. Set against that, coarse grids (8px and wider) fall below it and
# are missed; so is a printed halftone at 134. The semantic stage is the backstop
# for those, and the alternative — dropping to 300 to catch them — would leave a
# real dermoscopy image barely 2x clear of a refusal it could not override.
_MOIRE_MAX_PEAK_RATIO = float(os.environ.get("SKIN_GATE_MOIRE_RATIO", "400"))
_MOIRE_PX = 512

# --- "is that the size of a skin spot?" -------------------------------------
# A mole is millimetres across. A face at arm's length is not. Once the housing
# fixes the working distance the field of view is a constant, so a measured
# width becomes a real check rather than a display value.
#
# This stage is OFF unless the scale has actually been measured. `pixels_per_mm`
# defaults to 10.0, which is a placeholder nobody derived from optics — the
# "About 46 mm across" on the results screen was that placeholder, not a
# measurement. Refusing a photo on the strength of it would be inventing a
# reason. Two conditions gate it, both required:
#
#   * the capture came from the fixed-optics device camera, not an upload
#     (a web-app upload has no knowable scale at all), and
#   * SKIN_PIXELS_PER_MM has been set from a ruler photograph
#     (scripts/calibrate_scale.py).
#
# 20 mm is deliberately generous: the ABCDE "D" cue is 6 mm, large lesions reach
# 15 mm, and this is meant to catch a face or a forearm, not to second-guess a
# dermatologist.
_MAX_LESION_MM = float(os.environ.get("SKIN_GATE_MAX_LESION_MM", "20.0"))

# --- "is that a spot, or is that the light?" ---------------------------------
# The three checks below close the hole this gate shipped with: a photograph of
# ordinary bare skin reached the classifier and came back a confident "benign".
# Every earlier check answered yes to it — there is skin (1.00), there is one
# dark region not several (dominance 1.00), it is not a screen, and the region
# has Lab contrast of 9.6 against a floor of 5.0. What the segmenter had
# actually outlined was a shadow.
#
# Measured on synthetic frames matching tests/test_bare_skin.py — a skin field
# with a 0.55-1.15 illumination ramp, a soft shadow, hair, and a background
# band — against the lesion fixtures the suite already trusts:
#
#                                     on-skin   edge width   z
#   bare, lighting gradient              1.00         0.07   0.01
#   bare, gradient + hair                1.00         0.20   0.61
#   bare, soft shadow                    1.00         7.06   5.91
#   bare, shadow + hair                  1.00         6.17   2.44
#   bare, arm against a dark desk        0.01         1.14  10.96
#   bare, arm against a light wall       0.01         1.15   6.91
#   ---------------------------------------------------------------
#   lesion, pale (amelanotic)            1.00         1.08  12.69
#   lesion, single                       1.00         1.14  35.90
#   lesion, dark skin                    1.00         1.15  17.57
#   lesion, under heavy hair             1.00         1.32   9.16
#   lesion, blurred sigma=8              1.00         1.68 165.50
#
# No one of them catches every bare-skin frame and each is chosen for the case
# the other two cannot see: on-skin catches an outline that is not on the
# person, edge width catches shading, and z catches an outline that does not
# stand out from the skin's own variation. Each threshold sits at least 2x clear
# of the worst *correctly outlined* lesion above, which is the same margin rule
# _MOIRE_MAX_PEAK_RATIO was chosen under.
#
# All three are SOFT refusals. Every one of them is a judgement about the photo
# — its framing, its light, its contrast — not about the subject, and
# views/results_view.py draws the hard/soft line exactly there. The person who
# genuinely has a faint spot is the one person who can see the scanner is wrong,
# so "Check it anyway" must stay on screen for them.
#
# NOTE on the two frames these numbers do not flatter: a real lesion
# photographed under a strong gradient or a hard shadow scores 13.1 and 10.3 on
# edge width and IS refused. In both the segmenter had outlined the shading and
# not the mole (0.25-0.28 of the frame against the mole's 0.087), so the scan
# that used to "succeed" was measuring a shadow. "That looks like a shadow —
# even out the light and put the ring on the mole" is the better answer, and it
# is overridable.
_MIN_ON_SKIN = float(os.environ.get("SKIN_GATE_MIN_ON_SKIN", "0.50"))
_MAX_EDGE_WIDTH = float(os.environ.get("SKIN_GATE_MAX_EDGE_WIDTH", "4.0"))
_MIN_CONTRAST_Z = float(os.environ.get("SKIN_GATE_MIN_CONTRAST_Z", "1.5"))
# What the Settings "Check photos strictly" switch tightens them to. Still clear
# of every correctly outlined lesion above, but with less room — which is why it
# is opt-in and why its help text says it will reject some real lesions.
_MAX_EDGE_WIDTH_STRICT = float(os.environ.get("SKIN_GATE_MAX_EDGE_WIDTH_STRICT", "2.5"))
_MIN_CONTRAST_Z_STRICT = float(os.environ.get("SKIN_GATE_MIN_CONTRAST_Z_STRICT", "3.0"))


# Two values, and the difference decides whether a person can get past a
# refusal — worth the reader not having to grep for the spellings in use.
Severity = Literal["soft", "hard"]


@dataclass(frozen=True)
class FrameCheck:
    """What the gate saw. ``reasons`` are already in plain language."""

    is_lesion_photo: bool
    skin_fraction: float
    lesion_fraction: float
    lesion_contrast: float
    reasons: tuple[str, ...] = ()
    # A stable identifier for *why*, so the result screen can say something
    # specific without matching on the sentence. "NO SKIN SPOT FOUND" is a fine
    # headline for a wall and nonsense for a photograph of a face.
    code: str = ""
    # "soft" refusals offer "Check it anyway"; "hard" ones do not.
    #
    # Every refusal that predates this field is soft, and deliberately stays
    # soft: each one is a framing or contrast judgement that can be wrong about
    # an unusual real lesion, and the cost of being wrong there is a missed
    # melanoma. Hard is reserved for the refusals that are about the *subject*
    # rather than the photo's quality — this is not one spot, this is not skin
    # at a spot's scale — where forcing the scan through would put a verdict on
    # something the classifier has no business reading. Its three classes are
    # benign, pre-cancerous and malignant; there is no "not a lesion" among them,
    # so anything that reaches it gets one of those three whatever it is.
    severity: Severity = "soft"

    @property
    def has_skin(self) -> bool:
        return self.skin_fraction >= _SKIN_MIN

    @property
    def can_override(self) -> bool:
        return self.severity != "hard"


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
    ring = _ring_of(mask)
    if not ring.any():
        return 0.0
    lab = _lab(image_rgb)
    inside = lab[m].mean(axis=0)
    outside = lab[ring].mean(axis=0)
    return float(np.linalg.norm(inside - outside))


def skin_region(image_rgb: np.ndarray) -> np.ndarray:
    """:func:`skin_mask` with its holes filled — "where is the person's skin?"

    The distinction matters and getting it wrong the obvious way rejects every
    mole in the world. ``skin_mask`` classifies by *skin colour*, and a
    pigmented lesion is not skin-coloured: measured on the suite's own
    single-lesion fixture, the lesion outline overlaps the raw skin mask by
    **0.00**. Any "the spot must be on skin" rule written against the raw mask
    refuses 100% of dark lesions.

    Closing the mask and filling its external contours turns "pixels that look
    like skin" into "the area of the photo that is a person", so a lesion
    sitting inside skin counts as on-skin (1.00) while an arm photographed
    against a desk still reads the desk as not-skin.
    """
    # Downscaled first: this is compared against masks of every resolution via
    # _match_shape, and the 384px copy is what keeps it a couple of milliseconds
    # on a Pi rather than tens on a 12 MP capture.
    m = skin_mask(_work_size(image_rgb)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def _ring_of(mask: np.ndarray) -> np.ndarray:
    """The band of skin just outside ``mask`` — what a spot is compared against.

    One definition, used by :func:`lesion_contrast`, :func:`edge_width` and
    :func:`skin_variation` alike. They are compared against each other's
    thresholds, so a ring that meant something slightly different in each would
    make those comparisons meaningless.
    """
    m = mask > 0
    k = max(3, int(0.06 * max(m.shape[:2])) | 1)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1) > 0
    return ring & ~m


def mask_on_skin(mask: np.ndarray | None, skin_geometry: np.ndarray | None) -> float:
    """Share of the outlined spot that sits on skin, in [0, 1].

    Measured against :func:`skin_region` — the hole-filled one — for the reason
    given there: against the raw colour mask a real mole scores 0.00 and the
    check would refuse every lesion in the world.

    The obvious alternative, measuring the *ring* around the spot instead, was
    tried and abandoned: it scores 1.00 on everything. When the segmenter
    latches onto a dark background band the band runs off the frame edge, so
    the ring around it is the arm — all skin, no complaint. It is the outlined
    area itself that has to be on the person.

    Measured, on synthetic frames matching tests/test_bare_skin.py:

        every lesion fixture, dark or pale, centred or at the edge   1.00
        forearm photographed against a dark desk                     0.01
        forearm photographed against a light wall                    0.01

    Returns 1.0 — "no opinion" — when there is no geometry to compare against,
    so a caller that could not measure the skin never causes a refusal.
    """
    if skin_geometry is None or mask is None:
        return 1.0
    m = mask > 0
    if not m.any():
        return 1.0
    skin = _match_shape(skin_geometry, m.shape[:2])
    return float(np.count_nonzero(m & skin)) / float(np.count_nonzero(m))


def _match_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask, so the 384px skin geometry can
    be compared against a native-resolution lesion outline."""
    if mask.shape[:2] == tuple(shape):
        return mask > 0
    resized = cv2.resize(
        mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return resized > 0


def edge_width(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    """How fast the spot's edge happens, as a percentage of the frame diagonal.

    A percentage, not a pixel count, so the number means the same thing for a
    Pi capture and a 12 MP upload. ``_MAX_EDGE_WIDTH`` is on this scale.

    This is the measure that separates a shadow from a mole, and it is *not*
    contrast. Contrast asks how different the spot is, and a lighting gradient
    can be arbitrarily different and still be a gradient — which is exactly how
    a bare forearm reached the classifier and came back "benign" with a Lab
    contrast of 14.3 against a floor of 5.0.

    Width asks how *fast* the difference happens, and that is the one property
    shading cannot fake. A penumbra's width is set by the light source's angular
    size and spans a large part of the frame; a mole's border is set by pigment
    and stays narrow even when the photo is soft. Measured (see the table beside
    ``_MAX_EDGE_WIDTH``):

        pale (amelanotic) lesion            1.08
        single lesion                       1.14
        lesion + hair                       1.32
        lesion, blurred sigma=8             1.68   <- worst real lesion
        --------------------------------------
        bare forearm, shadow + hair         6.17   <- refused
        bare forearm, soft shadow           7.06   <- refused

    ``median`` along the boundary, never ``mean``: a hair crossing the outline
    is a huge local gradient and would make any frame look sharp-edged. That
    asymmetry is why this can only ever be a "this edge is far too wide"
    refusal, and never a "sharp enough, let it through" acceptance.
    """
    work = _work_size(image_rgb)
    m = _match_shape(mask, work.shape[:2])
    ring = _ring_of(m)
    if not m.any() or not ring.any():
        return 0.0

    lab_l = cv2.GaussianBlur(_lab(work)[:, :, 0], (0, 0), 2.0)
    drop = abs(float(lab_l[m].mean()) - float(lab_l[ring].mean()))
    if drop <= 0.0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    band = cv2.dilate(m.astype(np.uint8), k) - cv2.erode(m.astype(np.uint8), k)
    band = band > 0
    if not band.any():
        return 0.0
    gx = cv2.Scharr(lab_l, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lab_l, cv2.CV_32F, 0, 1)
    # Scharr's kernel sums to 32; dividing keeps the gradient in Lab-L per pixel
    # so the ratio below is a real pixel count rather than an arbitrary scale.
    slope = float(np.median(np.hypot(gx, gy)[band])) / 32.0
    if slope <= 1e-6:
        # No measurable slope anywhere on the boundary: the edge is infinitely
        # wide. Returned on the same percentage scale as every other path — an
        # earlier version returned a raw pixel count here, which still refused
        # the frame but poisoned the percentiles in the scan log and in
        # scripts/measure_gate_signals.py, i.e. the calibration set itself.
        return 100.0
    width_work = drop / slope
    return 100.0 * width_work / float(np.hypot(*work.shape[:2]))


def skin_variation(image_rgb: np.ndarray, mask: np.ndarray, skin_geometry: np.ndarray | None) -> float:
    """How much the skin in this frame varies on its own, in Lab units.

    The denominator for :func:`contrast_z`. Measured over skin *outside* the
    spot and its ring, so the lesion cannot inflate the figure it is about to be
    judged against. Returns 0.0 when there is not enough skin left to measure,
    which the caller reads as "no opinion".
    """
    work = _work_size(image_rgb)
    m = _match_shape(mask, work.shape[:2])
    ring = _ring_of(m)
    elsewhere = ~(m | ring)
    if skin_geometry is not None:
        elsewhere &= _match_shape(skin_geometry, work.shape[:2])
    if np.count_nonzero(elsewhere) < 64:
        return 0.0
    lab = _lab(work)[elsewhere]
    return float(np.sqrt(np.mean(lab.std(axis=0) ** 2)))


def contrast_z(contrast: float, sigma_skin: float) -> float:
    """Lesion contrast measured in units of the skin's own variation.

    The absolute Lab floor is what let a bare forearm through: on a frame with a
    lighting gradient the outlined region's contrast is 5.6, over the floor of
    5.0, while the skin around it varies by 9.07 — so the "spot" does not stand
    out from that skin at all, and scores 0.61 here.

    Armed by default, at a floor far below any lesion measured here. It was
    planned as a strict-mode-only check on the expectation that the margin would
    be thin; measuring it on frames with a real lighting gradient showed the
    opposite — bare skin lands at 0.01-0.61 while the weakest lesion that
    reaches this check sits at 9.16 — and it is the only one of the three that
    catches a plain gradient with no shadow and no background in shot. Refusing
    that by default is what the device is for. ``_MIN_CONTRAST_Z`` carries the
    numbers and ``_MIN_CONTRAST_Z_STRICT`` the tightened value.

    ``0.0`` sigma means "not measurable" and returns infinity — no opinion,
    never a refusal.
    """
    if sigma_skin <= 0.0:
        return float("inf")
    return contrast / sigma_skin


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
    pct = (_feature_stats() or {}).get("train_distance_percentiles") or {}
    furthest = pct.get("100")
    if furthest:
        return float(furthest) * _OOD_MARGIN
    p99 = pct.get("99")
    # No 100th percentile in an older stats file: approximate the same intent
    # rather than falling back to the tight p99 x 1.1 line.
    return float(p99) * 2.0 if p99 else None


def ood_stage_available() -> bool:
    """True when stage 3 can actually run (``models/feature_stats.json`` exists).

    Callers use this to decide whether the conv features are worth computing at
    all. That matters because those features cost a second full forward pass:
    without this predicate, "skip the CAM to save time" would silently disable a
    safety gate, and with it the cost is paid only when the gate is real.
    """
    return _ood_threshold() is not None


def ood_report(features: np.ndarray) -> tuple[float | None, float | None]:
    """``(distance, threshold)`` for logging, either of which may be ``None``.

    Exposed so every scan can record where it actually landed. Without a number
    in the log there is no way to tell a correctly-refused face from a threshold
    set too tight for this camera, and no way to tune SKIN_GATE_MAX_OOD except
    by guessing.
    """
    return feature_distance(features), _ood_threshold()


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
    return FrameCheck(
        False, skin, 0.0, 0.0, ("No skin was found in this photo.",), code="no_skin"
    ), skin


def spot_signals(
    image_rgb: np.ndarray,
    mask: np.ndarray | None,
    *,
    skin_geometry: np.ndarray | None = None,
) -> dict[str, float]:
    """Every number the content checks are decided on, measured once.

    Returned whether or not the frame is refused, and logged on every scan by
    ``services.pipeline``. The thresholds in this module were set from synthetic
    frames because no photographs of the failure existed to set them from; the
    log is how real captures replace that. Do not make this quietly skip work
    when the frame is going to pass — a refusal that never happens is exactly
    the measurement worth having.
    """
    if mask is None or not (mask > 0).any():
        return {"on_skin": 1.0, "edge_width": 0.0, "contrast": 0.0, "sigma_skin": 0.0, "z": float("inf")}

    # Everything is measured on the 384px working copy, and that is load-bearing
    # rather than a speed choice. These signals are compared against one set of
    # thresholds from two places — quick_reject, which holds a 384px frame, and
    # check_spot, which holds the full capture — and Lab contrast measured at
    # native resolution is not the same number as the same contrast measured
    # after an area-average downscale. Normalising here is what makes the two
    # paths give one photograph one answer.
    work = _work_size(image_rgb)
    m = _match_shape(mask, work.shape[:2]).astype(np.uint8) * 255
    if not (m > 0).any():
        # The outline vanished in the downscale: a spot thinner than one working
        # pixel. "Too small to read" is check_spot's business, not this one's.
        return {"on_skin": 1.0, "edge_width": 0.0, "contrast": 0.0, "sigma_skin": 0.0, "z": float("inf")}

    contrast = lesion_contrast(work, m)
    sigma = skin_variation(work, m, skin_geometry)
    return {
        "on_skin": mask_on_skin(m, skin_geometry),
        "edge_width": edge_width(work, m),
        "contrast": contrast,
        "sigma_skin": sigma,
        "z": contrast_z(contrast, sigma),
    }


def _content_refusal(
    signals: dict[str, float], *, skin: float, frac: float, strict: bool
) -> FrameCheck | None:
    """The three "is that a spot, or is that the light?" checks, or None.

    Shared by :func:`check_spot` and :func:`quick_reject` so the cheap
    pre-segmentation path and the full path cannot drift apart — and so bare
    skin is still refused *before* the expensive stages run. See
    tests/test_gate_cost.py: reaching a refusal used to cost 107 seconds on a
    12 MP capture, and putting these checks only in the full path would bring
    that back.
    """
    max_edge = _MAX_EDGE_WIDTH_STRICT if strict else _MAX_EDGE_WIDTH
    min_z = _MIN_CONTRAST_Z_STRICT if strict else _MIN_CONTRAST_Z

    if signals["on_skin"] < _MIN_ON_SKIN:
        return FrameCheck(
            False,
            skin,
            frac,
            signals["contrast"],
            ("What is inside the ring is not on skin.",),
            code="off_skin",
        )
    if signals["edge_width"] > max_edge:
        return FrameCheck(
            False,
            skin,
            frac,
            signals["contrast"],
            ("The edge of what is inside the ring fades away like a shadow.",),
            code="soft_edge",
        )
    if signals["z"] < min_z:
        return FrameCheck(
            False,
            skin,
            frac,
            signals["contrast"],
            ("This looks like plain skin with no clear spot on it.",),
            code="plain_skin",
        )
    return None


def check_spot(
    image_rgb: np.ndarray,
    mask: np.ndarray | None,
    *,
    skin: float = 1.0,
    mask_is_a_guess: bool = False,
    skin_geometry: np.ndarray | None = None,
    strict: bool = False,
    signals: dict[str, float] | None = None,
) -> FrameCheck:
    """Stage 2 alone: is there a distinct spot on skin already known to be there?

    ``skin`` is passed in by callers that already measured it (the pipeline),
    so the mask and colour work is not repeated once per scan.

    ``mask_is_a_guess`` is ``segment_or_fallback``'s flag: the outline is a
    circle drawn in the middle of the frame because nothing plausible was found.
    Measuring contrast on it compares skin against skin, so it must not be
    allowed to answer "there is a spot here" — before this flag existed the
    "no spot" branch below could not be reached through the pipeline at all.

    ``skin_geometry`` is :func:`skin_region` for this frame. It is keyword-only
    and defaults to ``None`` — "no opinion" — so every existing caller keeps
    working and the on-skin check simply does not run for them. Note that "no
    opinion" is not free: without it a frame whose outline is off the person is
    still refused, but as ``plain_skin`` ("no clear spot on this skin") rather
    than ``off_skin`` ("move so only skin fills the ring"), which is the less
    useful of the two sentences. Pass it wherever it is known.

    ``signals`` lets a caller hand in :func:`spot_signals` it has already
    measured. ``services.pipeline`` does, because it logs them on every scan and
    measuring the same frame twice is real time on a Pi.
    """
    reasons: list[str] = []
    if mask_is_a_guess:
        reasons.append("No spot could be picked out on the skin.")
        return FrameCheck(False, skin, 0.0, 0.0, tuple(reasons), code="no_spot")
    if mask is None or not (mask > 0).any():
        reasons.append("No spot could be picked out on the skin.")
        return FrameCheck(False, skin, 0.0, 0.0, tuple(reasons), code="no_spot")

    frac = float(np.count_nonzero(mask > 0)) / float(mask.size)
    contrast = lesion_contrast(image_rgb, mask)

    code = ""
    if frac < _LESION_MIN_FRAC:
        reasons.append("The spot is too small in the frame. Move the camera closer.")
        code = "too_small"
    elif frac > _LESION_MAX_FRAC or _touches_all_borders(mask):
        reasons.append("The spot fills the whole photo. Move the camera back a little.")
        code = "fills_frame"
    elif contrast < _MIN_CONTRAST:
        reasons.append("This looks like plain skin with no clear spot on it.")
        code = "plain_skin"

    if reasons:
        return FrameCheck(False, skin, frac, contrast, tuple(reasons), code=code)

    # Size, framing and raw contrast all said yes. So did they for a photograph
    # of a bare forearm, which is why these run last rather than not at all.
    if signals is None:
        signals = spot_signals(image_rgb, mask, skin_geometry=skin_geometry)
    content = _content_refusal(signals, skin=skin, frac=frac, strict=strict)
    if content is not None:
        return content

    return FrameCheck(True, skin, frac, contrast, (), code="")


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


def tone_spread(image_rgb: np.ndarray) -> float:
    """How much lighter-to-darker range the frame actually contains.

    Exposure-normalised first, for the same reason ``skin_mask`` does it: a dim
    capture has its whole tonal range compressed, so a real lesion shot in poor
    indoor light scores like bare skin and would be refused as "no clear spot".

    Sizes the frame itself. It used to take an already-downscaled copy and every
    caller had to remember that; a full-resolution frame handed to it by mistake
    would have scored differently and nothing would have said so.
    """
    lab_l = _lab(_normalise_exposure(_work_size(image_rgb)))[:, :, 0]
    smooth = cv2.medianBlur(lab_l, 5).astype(np.float32)
    return float(np.percentile(smooth, 99.5) - np.percentile(smooth, 0.5))


def dark_structure(image_rgb: np.ndarray) -> tuple[int, float, float]:
    """``(blob count, dominance, contrast)`` for the dark regions of a frame.

    *Dominance* is the largest dark blob's share of all dark-blob area: 1.0 when
    one region holds everything, falling toward 1/n as several share it.
    *Contrast* is the Lab distance between the dark set and everything else, and
    exists to say whether there is any real structure here at all.

    Only the largest connected component survives ``services.segmentation``, so
    by the time the existing stage-2 check sees a mask the "several regions"
    signal has already been thrown away. This measures it before that happens.
    """
    work = _work_size(image_rgb)
    normalised = _normalise_exposure(work)
    lab_l = cv2.cvtColor(normalised, cv2.COLOR_RGB2LAB)[:, :, 0]
    # Median blur first: per-pixel capture noise otherwise fragments one blob
    # into dozens and would make every frame look like a scene.
    smooth = cv2.medianBlur(lab_l, 5)
    _, dark = cv2.threshold(255 - smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=1)

    is_dark = dark > 0
    if not is_dark.any() or is_dark.all():
        return 0, 1.0, 0.0
    lab_f = _lab(normalised)
    contrast = float(np.linalg.norm(lab_f[is_dark].mean(axis=0) - lab_f[~is_dark].mean(axis=0)))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    total = float(dark.size)
    areas = sorted(
        (
            int(a)
            for a in stats[1:, cv2.CC_STAT_AREA]
            if _STRUCTURE_MIN_BLOB_FRAC * total <= a <= _STRUCTURE_MAX_BLOB_FRAC * total
        ),
        reverse=True,
    )
    if not areas:
        return 0, 1.0, contrast
    return len(areas), areas[0] / float(sum(areas)), contrast


def check_structure(image_rgb: np.ndarray, *, skin: float = 1.0) -> FrameCheck | None:
    """Refuse a frame holding several separate dark marks rather than one spot."""
    blobs, dominance, contrast = dark_structure(image_rgb)
    if (
        blobs >= _STRUCTURE_MIN_BLOBS
        and dominance < _STRUCTURE_MAX_DOMINANCE
        and contrast >= _STRUCTURE_MIN_CONTRAST
    ):
        return FrameCheck(
            False,
            skin,
            0.0,
            contrast,
            ("This photo has several dark marks in it, not one spot.",),
            code="structure",
            severity="hard",
        )
    return None


def check_scale(
    mask: np.ndarray | None, pixels_per_mm: float | None, *, skin: float = 1.0
) -> FrameCheck | None:
    """Refuse a spot measuring far wider than any skin lesion.

    ``pixels_per_mm`` of ``None`` means the scale is not trustworthy for this
    capture, and the check does not run. That is the normal case for the web
    app, where an uploaded photo carries no scale at all.
    """
    if mask is None or not pixels_per_mm or pixels_per_mm <= 0:
        return None
    from services.abcde import diameter_mm

    mm = diameter_mm(mask, pixels_per_mm)
    if mm <= _MAX_LESION_MM:
        return None
    return FrameCheck(
        False,
        skin,
        0.0,
        0.0,
        (f"What is in the ring measures about {mm:.0f} mm across.",),
        code="too_large",
        severity="hard",
    )


def _periodicity(image_rgb: np.ndarray) -> float:
    """Sharpest repeating pattern in the frame, relative to its typical detail.

    Runs on a native-resolution centre crop rather than the working copy: the
    downscale that makes the rest of the gate cheap is exactly what removes the
    high frequencies a screen grid lives in.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape[:2]
    if min(h, w) < _MOIRE_PX:
        crop = cv2.resize(gray, (_MOIRE_PX, _MOIRE_PX), interpolation=cv2.INTER_AREA)
    else:
        y, x = (h - _MOIRE_PX) // 2, (w - _MOIRE_PX) // 2
        crop = gray[y : y + _MOIRE_PX, x : x + _MOIRE_PX]
    # Hann window, or the crop's own edges ring across the whole spectrum.
    window = np.outer(np.hanning(_MOIRE_PX), np.hanning(_MOIRE_PX)).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((crop - crop.mean()) * window)))
    centre = _MOIRE_PX // 2
    rows, cols = np.ogrid[:_MOIRE_PX, :_MOIRE_PX]
    radius = np.hypot(rows - centre, cols - centre)
    # Skip the centre (overall brightness) and the corners (beyond Nyquist).
    band = (radius > _MOIRE_PX * 0.05) & (radius < _MOIRE_PX * 0.48)
    values = spectrum[band]
    median = float(np.median(values)) or 1e-6
    return float(values.max() / median)


def check_screen(image_rgb: np.ndarray, *, skin: float = 1.0) -> FrameCheck | None:
    """Refuse a photograph of a display or a printed picture."""
    if _periodicity(image_rgb) > _MOIRE_MAX_PEAK_RATIO:
        return FrameCheck(
            False,
            skin,
            0.0,
            0.0,
            ("This looks like a photo of a screen or a printed picture.",),
            code="screen",
            severity="hard",
        )
    return None


def quick_reject(
    image_rgb: np.ndarray,
    *,
    skin: float = 1.0,
    skin_geometry: np.ndarray | None = None,
    strict: bool = False,
    signals_out: dict[str, float] | None = None,
) -> FrameCheck | None:
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

    * **"plain skin, no clear spot"** — decided from :func:`tone_spread`, which
      needs no mask at all. That is the point: the obvious implementation asks
      ``segment_safe`` for a mask, but that helper substitutes a *centre circle*
      when no candidate is plausible, and measuring contrast on that circle
      compares skin against skin for any off-centre lesion. It would refuse
      precisely the frames the expensive path exists to rescue.
    * **"fills the whole photo"** — geometry, which downscaling does not change.
      Only trusted when the cheap mask is in band; a degenerate mask means no
      opinion.
    * **the three content checks** — not on skin, a shadow's soft edge, or no
      contrast against the skin's own variation. Same trust rule as above. These
      are what actually stop a photograph of a bare forearm; the tone-spread
      test below turned out to be dead on any frame with a lighting gradient in
      it, which is every real photograph of an arm.

    **"too small in the frame" is never used here.** A small lesion is what the
    enhancement and GrabCut exist to find, so the coarse pass must not reject
    one. (A speck below ``_LESION_MIN_FRAC`` can still fall under the spread
    threshold and be refused as "plain skin" — the full path refuses it too,
    just with a different sentence.)

    Returning ``None`` means "no opinion" — the full path runs and decides.
    """
    work = _work_size(image_rgb)

    if tone_spread(work) < _PRECHECK_MIN_SPREAD:
        return FrameCheck(
            False,
            skin,
            0.0,
            0.0,
            ("This looks like plain skin with no clear spot on it.",),
            code="plain_skin",
        )

    # Both of these are cheap and neither needs a mask, so they run before any
    # segmentation work. Order matters only for which sentence the user reads:
    # the spread check above claims genuinely flat frames first, so "several
    # dark marks" cannot be said about bare skin.
    structure = check_structure(work, skin=skin)
    if structure is not None:
        return structure

    # Native resolution, not `work` — see _periodicity.
    screen = check_screen(image_rgb, skin=skin)
    if screen is not None:
        return screen

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
            code="fills_frame",
        )

    # The three content checks, on the coarse mask and under the same trust rule
    # the fills-frame branch above already applies: only when the cheap mask is
    # in band, because a degenerate one is not evidence of anything.
    #
    # They run here as well as in check_spot on purpose. A frame of bare skin is
    # the case that pays most for reaching a refusal the slow way — see this
    # function's docstring and tests/test_gate_cost.py — and refusing it only
    # after enhancement and GrabCut would put those 107 seconds back.
    signals = spot_signals(work, mask, skin_geometry=skin_geometry)
    # Handed back rather than returned, because the return value is already "the
    # refusal, or None to keep going" and both of those need to be loggable.
    # Without this the frames that ARE refused here — the interesting ones —
    # were the only frames a scan never recorded any measurement for, which
    # would have left the calibration log recording nothing but the passes.
    if signals_out is not None:
        signals_out.update(signals)
    return _content_refusal(signals, skin=skin, frac=frac, strict=strict)


def check_frame(image_rgb: np.ndarray, mask: np.ndarray | None) -> FrameCheck:
    """Both content stages in order. ``mask`` comes from ``services.segmentation``.

    Measures the skin geometry itself rather than leaving it at ``None``. It
    could not simply be omitted: without it an outline sitting on a desk beside
    the arm is still refused, but the reason reaching the screen is "no clear
    spot on this skin" instead of "move so only skin fills the ring" — the same
    refusal wearing the wrong sentence, which is the exact failure
    ``verdict._NO_LESION_COPY`` exists to prevent.
    """
    no_skin, skin = check_skin(image_rgb)
    if no_skin is not None:
        return no_skin
    return check_spot(image_rgb, mask, skin=skin, skin_geometry=skin_region(image_rgb))
