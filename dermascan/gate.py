"""Is this a readable photo of one spot on skin? Four checks, a few milliseconds each.

The classifier has three labels - benign, pre_cancerous, malignant - and no
"not a lesion". Its probabilities always sum to 1, so a photo of a wall comes
back as a confident "benign". Something has to say "this is not a spot on skin"
BEFORE the model is asked, and that is this file.

    1. skin      Is there skin in the frame at all?      (colour)
    2. readable  Is the photo sharp and lit?             (focus, brightness)
    3. spot      Is there one distinct mark on it?       (outline, compactness, contrast)
    4. edge      Does the mark have a real edge?         (a mole stops; a shadow fades)

Every check runs on a small copy of the photo. Every threshold lives in
config.py. A refusal carries a `code` that verdict.py turns into plain words,
and `can_override`: checks 3 and 4 can be wrong about a faint or unusual mark,
so the visitor may ask for it to be read anyway; checks 1 and 2 cannot be
argued with (no skin, or a photo nobody could read).

What was deliberately left out, after an earlier version grew to 1,100 lines:
screen/moire detection (the students demo with photos shown on a phone, which
it refused), contrast-variation and on-skin-geometry checks (tuned on synthetic
frames, never on this camera), and GrabCut (20 s on a Pi).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from dermascan import config

RefusalCode = Literal["no_skin", "too_blurry", "too_dark", "too_bright", "no_spot", "fills_frame", "plain_skin", "soft_edge"]


@dataclass(frozen=True)
class Refusal:
    code: RefusalCode
    can_override: bool


@dataclass(frozen=True)
class GateReport:
    refusal: Refusal | None  # None means the photo passed
    measured: dict[str, float]  # every number the checks decided on, for the staff panel and the log
    # The spot's outline on the GATE_WORK_PX copy, once check 3 has drawn one.
    # signs.py measures A, B and C on it, so nothing outlines the spot twice.
    outline: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def passed(self) -> bool:
        return self.refusal is None


# --- measuring ------------------------------------------------------------------


def shrink(rgb: np.ndarray, longest: int = config.GATE_WORK_PX) -> np.ndarray:
    """Reduce to `longest` px on the long edge. Never enlarges."""
    h, w = rgb.shape[:2]
    if max(h, w) <= longest:
        return rgb
    s = longest / float(max(h, w))
    return cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _normalise_exposure(rgb: np.ndarray) -> np.ndarray:
    """Lift a dim frame before judging colour (never darkens; gain capped at 4x)."""
    v = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2].mean())
    target = config.SKIN_EXPOSURE_TARGET
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
    return float(skin_mask(shrink(rgb)).mean())


def focus_score(rgb: np.ndarray) -> float:
    """Sharpness independent of size and exposure: Laplacian variance at a fixed size
    after a contrast stretch. A dim-but-sharp photo must not read as blurry."""
    gray = cv2.cvtColor(shrink(rgb, config.FOCUS_WORK_PX), cv2.COLOR_RGB2GRAY)
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
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _clean_threshold(gray: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Otsu's threshold (on the `valid` pixels only, if given), then small holes and specks removed."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sample = blur[valid] if valid is not None and valid.any() else blur.ravel()
    t, _ = cv2.threshold(sample.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = ((blur > t) & (valid if valid is not None else True)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)


def _otsu_blob(gray: np.ndarray) -> np.ndarray:
    return _largest_component(_clean_threshold(gray))


def _central_component(binary: np.ndarray) -> np.ndarray | None:
    """The largest blob that keeps clear of the frame edge, or None if there is none big enough."""
    m = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    h, w = m.shape
    best, best_area = None, config.SPOT_MIN_FRACTION * h * w
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h and area >= best_area:
            best, best_area = i, area
    return None if best is None else labels == best


def _without_hair(l_ch: np.ndarray) -> np.ndarray:
    """Lightness with thin dark lines (hairs) filled in by the skin around them.

    A morphological close removes any dark feature narrower than the kernel and
    leaves anything wider - a mole - in place. Without it, hairs crossing a mole
    become the "outline": a thin web the compactness check rightly calls a scatter.
    """
    k = max(3, int(config.HAIR_KERNEL_FRACTION * max(l_ch.shape)) | 1)
    return cv2.morphologyEx(l_ch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _dark_spot(l_ch: np.ndarray) -> np.ndarray:
    """The dark region, preferring a spot in the field over dark corners at the frame edge.

    Dermoscopy photos - and the Pi's imaging cone - have a dark vignette around
    the field. Otsu's threshold then splits "vignette" from "everything else" and
    the outline lands on the corners. So: if the dark region found touches the
    edge, threshold again with those edge-touching dark pixels left out, and take
    a blob that keeps clear of the edge. Only if there is none is the edge blob
    kept - a spot that really is bigger than the frame (the caller refuses it).
    """
    dark_all = _clean_threshold(255 - l_ch) > 0
    first = _largest_component(dark_all)
    if not _touches_border(first):
        return first
    central = _central_component(dark_all)
    if central is not None and _looks_like_a_spot(central):
        return central
    n, labels = cv2.connectedComponents(dark_all.astype(np.uint8), connectivity=8)
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    vignette = np.isin(labels, edge_ids[edge_ids > 0])
    if vignette.mean() < 0.9:
        retry = _central_component(_clean_threshold(255 - l_ch, valid=~vignette) > 0)
        if retry is not None and _looks_like_a_spot(retry):
            return retry
    return first


def solidity(mask: np.ndarray) -> float:
    """Blob area over its convex hull area: 1.0 for a filled circle, low for a scatter."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
    return float(np.count_nonzero(mask)) / hull_area if hull_area > 0 else 0.0


def _looks_like_a_spot(mask: np.ndarray) -> bool:
    frac = float(mask.mean())
    return config.SPOT_MIN_FRACTION <= frac <= config.SPOT_MAX_FRACTION and solidity(mask) >= config.SPOT_MIN_SOLIDITY


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def outline_spot(rgb_small: np.ndarray) -> np.ndarray:
    """A boolean outline of the spot in a small frame.

    Moles are dark, so the dark region (Otsu's threshold on Lab lightness, hairs
    filled in first, dark frame corners set aside) is the outline whenever it
    looks like a spot. Only when it does not is the LIGHT
    region tried, for a pale mark on dark skin; it must be a compact blob clear of
    the frame edge, or it is just the skin around a dark region. Otherwise the
    dark region is returned so the caller can say why it was not a spot.
    """
    l_ch = _without_hair(cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)[:, :, 0])
    dark = _dark_spot(l_ch)
    if _looks_like_a_spot(dark):
        return dark
    light = _otsu_blob(l_ch)
    if _looks_like_a_spot(light) and not _touches_border(light):
        return light
    return dark


def ring_around(mask: np.ndarray) -> np.ndarray:
    """The band of skin just outside the spot: what the spot is compared against."""
    k = max(3, int(config.SPOT_RING_FRACTION * max(mask.shape)) | 1)
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~mask


def spot_contrast(rgb_small: np.ndarray, mask: np.ndarray) -> float:
    """CIE Lab distance between the spot and the ring of skin around it."""
    ring = ring_around(mask)
    if not mask.any() or not ring.any():
        return 0.0
    lab = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    return float(np.linalg.norm(lab[mask].mean(axis=0) - lab[ring].mean(axis=0)))


def edge_width(rgb_small: np.ndarray, mask: np.ndarray) -> float:
    """How far the outline's edge is smeared, as a percentage of the frame diagonal.

    Contrast cannot tell a mole from a shadow: a shadow can be as dark as you
    like. What a shadow cannot do is stop sharply. Take the lightness drop
    between the spot and its ring, divide by the median steepness of lightness
    along the outline, and you get "how many pixels the edge takes". Median, not
    mean, so a hair crossing the outline does not fake a sharp edge. Returns 100
    (infinitely wide) when there is a drop but no slope at all.
    """
    ring = ring_around(mask)
    if not mask.any() or not ring.any():
        return 0.0
    lab_l = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    lab_l = cv2.GaussianBlur(lab_l, (0, 0), 2.0)
    drop = abs(float(lab_l[mask].mean()) - float(lab_l[ring].mean()))
    m8 = mask.astype(np.uint8)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    outline = (cv2.dilate(m8, k7) - cv2.erode(m8, k7)) > 0
    if drop <= 0.0 or not outline.any():
        return 0.0
    gx = cv2.Scharr(lab_l, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lab_l, cv2.CV_32F, 0, 1)
    slope = float(np.median(np.hypot(gx, gy)[outline])) / 32.0  # Scharr kernels sum to 32
    if slope <= 1e-6:
        return 100.0
    return 100.0 * (drop / slope) / float(np.hypot(*mask.shape))


# --- the gate -------------------------------------------------------------------


def check(rgb: np.ndarray) -> GateReport:
    """Run the four checks in order and stop at the first that fails.

    Order is the order of useful advice: a photo of a wall also fails the focus
    check, and "hold still" is useless advice for it.
    """
    measured: dict[str, float] = {}

    outline: np.ndarray | None = None

    def refuse(code: RefusalCode, can_override: bool) -> GateReport:
        return GateReport(Refusal(code, can_override), measured, outline)

    # 1. Is there skin?
    measured["skin_fraction"] = skin_fraction(rgb)
    if measured["skin_fraction"] < config.SKIN_MIN_FRACTION:
        return refuse("no_skin", False)

    # 2. Can it be read?
    measured["focus"] = focus_score(rgb)
    measured["brightness"] = brightness(rgb)
    if measured["focus"] < config.FOCUS_MIN:
        return refuse("too_blurry", False)
    if measured["brightness"] < config.BRIGHTNESS_MIN:
        return refuse("too_dark", False)
    if measured["brightness"] > config.BRIGHTNESS_MAX:
        return refuse("too_bright", False)

    # 3. Is there one spot on it?
    small = shrink(rgb)
    mask = outline = outline_spot(small)
    measured["spot_fraction"] = float(mask.mean())
    measured["spot_solidity"] = solidity(mask)
    measured["spot_contrast"] = spot_contrast(small, mask)
    if measured["spot_fraction"] < config.SPOT_MIN_FRACTION:
        return refuse("no_spot", True)
    if measured["spot_fraction"] > config.SPOT_MAX_FRACTION:
        return refuse("fills_frame", True)
    if measured["spot_solidity"] < config.SPOT_MIN_SOLIDITY or measured["spot_contrast"] < config.SPOT_MIN_CONTRAST:
        return refuse("plain_skin", True)

    # 4. Does it have a real edge?
    measured["edge_width"] = edge_width(small, mask)
    if measured["edge_width"] > config.SPOT_MAX_EDGE_WIDTH_PCT:
        return refuse("soft_edge", True)
    return GateReport(None, measured, outline)
