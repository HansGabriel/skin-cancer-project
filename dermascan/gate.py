"""Is this a readable photo of one spot on skin? Three checks, each a few milliseconds.

The classifier has three labels - benign, pre_cancerous, malignant - and no
"not a lesion". Its probabilities always sum to 1, so a photo of a wall comes
back as a confident "benign". Something has to say "this is not a spot on skin"
BEFORE the model is asked, and that is this file.

    1. skin        Is there skin in the frame at all?      (colour)
    2. readable    Is the photo sharp and lit?             (focus, brightness)
    3. spot        Is there one distinct mark on it?       (outline, compactness, contrast)
    4. edge        Does the mark have a real edge?         (a mole stops; a shadow fades)

Every check runs on a 384 px copy of the photo. Every threshold lives in
config.py. A refusal carries a `code` the verdict turns into plain words, and
`can_override`: the spot check can be wrong about a faint or unusual mark, so
the visitor may ask for it to be read anyway; the other two cannot be argued
with (no skin, or a photo nobody could read).

What was deliberately left out, after an earlier version grew to 1,100 lines:
screen/moire detection (the students demo with photos shown on a phone, which
this refused), shadow-edge and contrast-variation checks (tuned on synthetic
frames, never on this camera), and GrabCut (20 s on a Pi). Fewer checks that
are each easy to explain beat many checks nobody can calibrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dermascan import config


@dataclass(frozen=True)
class Refusal:
    code: str  # no_skin | too_blurry | too_dark | too_bright | no_spot | fills_frame | plain_skin | soft_edge
    can_override: bool
    measured: dict[str, float]  # what the check saw, for the staff panel and the log


@dataclass(frozen=True)
class GateReport:
    refusal: Refusal | None
    measured: dict[str, float]  # skin_fraction, focus, brightness, spot_fraction, spot_contrast
    spot_mask: np.ndarray | None  # 384 px outline of the spot, when one was found

    @property
    def passed(self) -> bool:
        return self.refusal is None


# --- helpers --------------------------------------------------------------------


def shrink(rgb: np.ndarray, longest: int = config.GATE_WORK_PX) -> np.ndarray:
    """Reduce to `longest` px on the long edge. Never enlarges."""
    h, w = rgb.shape[:2]
    if max(h, w) <= longest:
        return rgb
    s = longest / float(max(h, w))
    return cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _normalise_exposure(rgb: np.ndarray, target: float = 110.0) -> np.ndarray:
    """Lift a dim frame before judging colour: JPEG crushes colour in low light,
    so a real lesion shot in a dim hall used to read as "no skin"."""
    v = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2].mean())
    if v <= 1.0 or v >= target:
        return rgb
    return cv2.convertScaleAbs(rgb, alpha=min(target / v, 4.0), beta=0)


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that look like human skin, at any melanin level.

    YCrCb chroma carries the tone-independent part; the HSV bounds only exclude
    greys and blues (walls, screens, denim) that fall inside the chroma box.
    Cb ceiling is 140, not the textbook 127, because polarised dermoscopy renders
    skin pink-violet (Cb 133-138). S >= 30 stops JPEG-grey desks reading as skin.
    V >= 20 (not 60) is what lets dark skin through under kiosk lighting.
    """
    rgb = _normalise_exposure(rgb)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.int16)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    chroma = (cr >= 136) & (cr <= 177) & (cb >= 77) & (cb <= 140)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue = ((h <= 27) | (h >= 158)) & (s >= 30) & (s <= 210) & (v >= 20)
    return chroma & hue


def skin_fraction(rgb: np.ndarray) -> float:
    m = skin_mask(shrink(rgb))
    return float(np.count_nonzero(m)) / float(m.size)


def focus_score(rgb: np.ndarray) -> float:
    """Sharpness independent of size and exposure: Laplacian variance at <=512 px
    after a contrast stretch. A dim-but-sharp photo must not read as blurry."""
    gray = cv2.cvtColor(shrink(rgb, 512), cv2.COLOR_RGB2GRAY)
    if int(gray.max()) - int(gray.min()) > 4:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())


def brightness(rgb: np.ndarray) -> float:
    return float(cv2.cvtColor(shrink(rgb), cv2.COLOR_RGB2HSV)[:, :, 2].mean())


def _largest_component(binary: np.ndarray) -> np.ndarray:
    m = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _otsu_blob(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return _largest_component(th)


def solidity(mask: np.ndarray) -> float:
    """Blob area over its convex hull area: 1.0 for a filled circle, low for a scatter."""
    if not mask.any():
        return 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
    return float(np.count_nonzero(mask)) / hull_area if hull_area > 0 else 0.0


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _plausible(mask: np.ndarray) -> bool:
    frac = float(mask.mean())
    return config.SPOT_MIN_FRACTION <= frac <= config.SPOT_MAX_FRACTION and solidity(mask) >= config.SPOT_MIN_SOLIDITY


def outline_spot(rgb_small: np.ndarray) -> np.ndarray:
    """A boolean outline of the spot in a 384 px frame.

    Moles are dark, so the dark region (Otsu's threshold on Lab lightness) is
    the outline whenever it looks like a spot. Only when it does not - too big,
    too small, or a scatter - is the LIGHT region tried, for a pale mark on dark
    skin; it must be a compact blob clear of the frame edge, or it is just the
    skin around a dark region that filled the frame. Otherwise the dark region
    is returned so the caller can say why it was not a spot.
    """
    l_ch = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)[:, :, 0]
    dark = _otsu_blob(255 - l_ch)
    if _plausible(dark):
        return dark
    light = _otsu_blob(l_ch)
    if _plausible(light) and not _touches_border(light):
        return light
    return dark


def spot_contrast(rgb_small: np.ndarray, mask: np.ndarray) -> float:
    """CIE Lab distance between the spot and the ring of skin just outside it."""
    if not mask.any():
        return 0.0
    k = max(3, int(0.06 * max(mask.shape)) | 1)
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~mask
    if not ring.any():
        return 0.0
    lab = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    return float(np.linalg.norm(lab[mask].mean(axis=0) - lab[ring].mean(axis=0)))


def edge_width(rgb_small: np.ndarray, mask: np.ndarray) -> float:
    """How far the outline's edge is smeared, as a percentage of the frame diagonal.

    Contrast cannot tell a mole from a shadow: a shadow can be as dark as you
    like. What a shadow cannot do is stop sharply. We take the lightness drop
    between the inside and the ring outside, divide by the median steepness of
    the lightness along the outline, and get "how many pixels does the edge
    take". Median, not mean, so a hair crossing the outline does not fake a
    sharp edge. Returns 100 (infinitely wide) when there is no slope at all.
    """
    if not mask.any():
        return 0.0
    lab_l = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    lab_l = cv2.GaussianBlur(lab_l, (0, 0), 2.0)
    k = max(3, int(0.06 * max(mask.shape)) | 1)
    m8 = mask.astype(np.uint8)
    ring = cv2.dilate(m8, np.ones((k, k), np.uint8)).astype(bool) & ~mask
    if not ring.any():
        return 0.0
    drop = abs(float(lab_l[mask].mean()) - float(lab_l[ring].mean()))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    band = (cv2.dilate(m8, k7) - cv2.erode(m8, k7)) > 0
    if drop <= 0.0 or not band.any():
        return 0.0
    gx = cv2.Scharr(lab_l, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lab_l, cv2.CV_32F, 0, 1)
    slope = float(np.median(np.hypot(gx, gy)[band])) / 32.0  # Scharr kernels sum to 32
    if slope <= 1e-6:
        return 100.0
    return 100.0 * (drop / slope) / float(np.hypot(*mask.shape))


# --- the gate -------------------------------------------------------------------


def check(rgb: np.ndarray) -> GateReport:
    """Run the three checks in order and stop at the first that fails.

    Order is the order of useful advice: a photo of a wall also fails the focus
    check, and "hold still" is useless advice for it.
    """
    small = shrink(rgb)
    measured: dict[str, float] = {}

    # 1. Is there skin?
    skin = float(skin_mask(small).mean())
    measured["skin_fraction"] = skin
    if skin < config.SKIN_MIN_FRACTION:
        return GateReport(Refusal("no_skin", False, dict(measured)), measured, None)

    # 2. Can it be read?
    focus = focus_score(rgb)
    light = float(cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 2].mean())
    measured["focus"] = focus
    measured["brightness"] = light
    if focus < config.FOCUS_MIN:
        return GateReport(Refusal("too_blurry", False, dict(measured)), measured, None)
    if light < config.BRIGHTNESS_MIN:
        return GateReport(Refusal("too_dark", False, dict(measured)), measured, None)
    if light > config.BRIGHTNESS_MAX:
        return GateReport(Refusal("too_bright", False, dict(measured)), measured, None)

    # 3. Is there one spot on it?
    mask = outline_spot(small)
    frac = float(mask.mean())
    contrast = spot_contrast(small, mask)
    compact = solidity(mask)
    measured["spot_fraction"] = frac
    measured["spot_contrast"] = contrast
    measured["spot_solidity"] = compact
    if frac < config.SPOT_MIN_FRACTION:
        return GateReport(Refusal("no_spot", True, dict(measured)), measured, mask)
    if frac > config.SPOT_MAX_FRACTION:
        return GateReport(Refusal("fills_frame", True, dict(measured)), measured, mask)
    if compact < config.SPOT_MIN_SOLIDITY or contrast < config.SPOT_MIN_CONTRAST:
        return GateReport(Refusal("plain_skin", True, dict(measured)), measured, mask)

    # 4. Does it have a real edge?
    width = edge_width(small, mask)
    measured["edge_width"] = width
    if width > config.SPOT_MAX_EDGE_WIDTH_PCT:
        return GateReport(Refusal("soft_edge", True, dict(measured)), measured, mask)
    return GateReport(None, measured, mask)
