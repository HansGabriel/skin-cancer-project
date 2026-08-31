"""A photograph of ordinary bare skin must not come back "benign".

This is the gap this file closes, measured rather than argued. A frame of a
forearm under uneven light reached the classifier and got a verdict, because
every check the gate had was answering a different question:

    skin fraction        1.00  against a floor of 0.08     -> yes, skin
    dark-blob dominance  1.00  against a ceiling of 0.65   -> yes, one region
    periodicity            18  against a ceiling of 400    -> not a screen
    Lab contrast          9.6  against a floor of 5.0      -> yes, a spot

What the segmenter had outlined was a shadow. The classifier has three labels —
benign, pre_cancerous, malignant — and no "not a lesion" among them, so a
3-class softmax summing to 1 turned a shadow into a confident benign.

Three checks close it, and each is here for a case the other two cannot see:

    off_skin    the outlined thing is not on the person (a phone in the hand,
                an arm against a desk). Measured against the *hole-filled* skin
                region, because a pigmented mole is not skin-coloured and scores
                0.00 against the raw colour mask — the naive version of this
                check refuses every dark lesion in the world.
    soft_edge   the outline's edge fades over tens of pixels the way a penumbra
                does, instead of stopping in a few the way pigment does.
    plain_skin  the outline does not stand out from the variation the skin in
                this frame already has.

**The fixtures here are generated, not photographed.** No photographs of the
failure were available when this shipped, which is the single most important
thing to know about the thresholds in ``services/lesion_gate.py``: they were set
from frames like these, and ``services.pipeline`` logs every one of these
measurements on every scan precisely so that real captures can replace them.
These tests pin the *shape* of the failure. They are not evidence about a real
camera.

Half of these tests are the other direction and they matter more. This project
has already shipped a gate that refused 72% of genuine HAM10000 images while the
suite stayed green (see ``tests/test_gate_real_images.py``), and an unusual real
lesion turned away is the failure it cannot afford.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.contracts import ScanResult
from services.lesion_gate import (
    edge_width,
    mask_on_skin,
    skin_mask,
    skin_region,
)
from services.pipeline import run_pipeline
from services.segmentation import segment_or_fallback

_SEED = 20260831


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _skin(h: int = 900, w: int = 900, tone: tuple[int, int, int] = (196, 158, 132)) -> np.ndarray:
    rng = _rng()
    base = np.full((h, w, 3), tone, np.uint8)
    mottle = rng.integers(-14, 14, (h, w, 1)).repeat(3, axis=2)
    grain = rng.integers(-9, 9, (h, w, 3))
    return np.clip(base.astype(np.int16) + mottle + grain, 0, 255).astype(np.uint8)


def _gradient(img: np.ndarray, lo: float = 0.55, hi: float = 1.15) -> np.ndarray:
    """Uneven light across the frame — what an arm under a room lamp looks like.

    This is the specific thing that killed the original plain-skin check: it
    measured tone spread over the whole frame against a threshold calibrated on
    flat synthetic patches, and a ramp like this scores 43 against a threshold
    of 12. On any real photograph of a limb the check could never fire.
    """
    h, w = img.shape[:2]
    ramp = np.linspace(lo, hi, w, dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def _shadow(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    m = np.zeros(img.shape[:2], np.float32)
    cv2.circle(m, (430, 470), 300, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 90)
    return np.clip(out * (1 - 0.35 * m[:, :, None]), 0, 255).astype(np.uint8)


def _hairs(img: np.ndarray, n: int = 40) -> np.ndarray:
    rng = _rng()
    out = img.copy()
    for _ in range(n):
        x, y = (int(v) for v in rng.integers(0, 900, 2))
        cv2.line(
            out,
            (x, y),
            (x + int(rng.integers(-200, 200)), y + int(rng.integers(-200, 200))),
            (38, 30, 28),
            3,
        )
    return out


def _against_background(img: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """The arm does not fill the frame — a desk or a wall is in shot beside it."""
    out = img.copy()
    out[:, 700:] = color
    return out


def _lesion(base: np.ndarray | None = None) -> np.ndarray:
    img = _skin() if base is None else base.copy()
    cv2.circle(img, (450, 450), 150, (78, 62, 74), -1)
    return img


def _pale_lesion(base: np.ndarray | None = None) -> np.ndarray:
    """Amelanotic: barely different from the skin around it."""
    img = _skin() if base is None else base.copy()
    cv2.circle(img, (450, 450), 150, (214, 150, 150), -1)
    return img


def _lesion_near_the_edge() -> np.ndarray:
    img = _skin()
    cv2.circle(img, (180, 180), 140, (78, 62, 74), -1)
    return img


def _jpeg(rgb: np.ndarray) -> bytes:
    """Through the real capture path, never as a raw array.

    This repo's history is that raw-array measurement is what shipped two wrong
    thresholds: JPEG chroma quantisation moves the skin test, and JPEG grain
    moves the noise floor the screen test divides by. A gate number measured on
    an ndarray is not the number the device will see.
    """
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


class _Backend:
    """Records whether the classifier was reached at all."""

    backend_id = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, *args, **kwargs) -> ScanResult:
        self.calls += 1
        return ScanResult(
            label="benign",
            confidence=0.93,
            probs={"benign": 0.93, "pre_cancerous": 0.04, "malignant": 0.03},
            inference_ms=1,
        )

    def health(self) -> dict:
        return {"status": "ok"}


def _scan(rgb: np.ndarray, *, strict: bool = False):
    backend = _Backend()
    return run_pipeline(backend, _jpeg(rgb), pixels_per_mm=10.0, strict=strict), backend


# --------------------------------------------------------------- must refuse

_BARE = {
    "lighting gradient": _gradient(_skin()),
    "gradient and hair": _hairs(_gradient(_skin())),
    "a soft shadow": _shadow(_skin()),
    "shadow and hair": _hairs(_shadow(_skin())),
    "arm against a dark desk": _against_background(_gradient(_skin()), (40, 34, 30)),
    "arm against a light wall": _against_background(_gradient(_skin()), (238, 240, 244)),
    "dark skin under a gradient": _gradient(_skin(tone=(120, 86, 66))),
}


@pytest.mark.parametrize("name", sorted(_BARE))
def test_bare_skin_never_reaches_the_classifier(name) -> None:
    result, backend = _scan(_BARE[name])

    assert result["blocked"] is True, f"bare skin ({name}) was given a verdict"
    assert result.get("scan_result") is None
    assert backend.calls == 0, "the classifier was asked about a photo of nothing"


@pytest.mark.parametrize("name", sorted(_BARE))
def test_that_refusal_always_leaves_a_way_through(name) -> None:
    """Soft, every one of them.

    These are judgements about the *photo* — its light, its framing, its
    contrast — not about the subject, and views/results_view.py draws the
    hard/soft line exactly there. The person who genuinely has a faint spot the
    scanner cannot see is the one person who knows it is really there.
    """
    result, _ = _scan(_BARE[name])
    check = result.get("frame_check")

    assert getattr(check, "can_override", True) is True
    assert getattr(check, "severity", "soft") == "soft"


def test_an_arm_against_a_background_is_refused_for_the_right_reason() -> None:
    """Not merely refused — refused with the sentence that helps.

    "Move so only skin fills the ring" is useful here and "even out the light"
    is not, which is the whole reason these are separate codes.
    """
    result, _ = _scan(_BARE["arm against a dark desk"])

    assert result["frame_check"].code == "off_skin"
    assert result["verdict"].headline == "THE RING IS NOT ON SKIN"


def test_a_shadow_is_refused_for_the_right_reason() -> None:
    result, _ = _scan(_BARE["a soft shadow"])

    assert result["frame_check"].code == "soft_edge"
    assert result["verdict"].headline == "THAT LOOKS LIKE A SHADOW"


def test_bare_skin_is_refused_before_the_expensive_stages_run() -> None:
    """The refusal must stay cheap.

    A non-lesion frame is the case that pays most for reaching a refusal the
    slow way: its threshold candidates score out of band, which is the one
    condition that forces GrabCut to run at native resolution. Measured at
    4032x3024 that was 107 seconds of work to say no. tests/test_gate_cost.py
    pins the same guarantee for the checks that existed before these three.
    """
    result, _ = _scan(_BARE["a soft shadow"])
    stages = result.get("stage_ms", {})

    for expensive in ("enhance", "segment", "model", "abcde"):
        assert expensive not in stages, f"paid for {expensive} only to refuse the frame"


# ----------------------------------------------------------------- must pass
#
# These matter more than the ones above. A bare forearm given a verdict is an
# embarrassment; a melanoma turned away is the thing this project cannot afford.

_LESIONS = {
    "a plain mole": _lesion(),
    "a pale (amelanotic) mole": _pale_lesion(),
    "a mole on dark skin": _lesion(_skin(tone=(120, 86, 66))),
    "a mole under heavy hair": _hairs(_lesion()),
    "a mole near the frame edge": _lesion_near_the_edge(),
}


@pytest.mark.parametrize("name", sorted(_LESIONS))
def test_a_real_lesion_is_still_outlined_on_skin(name) -> None:
    """The on-skin check, in the direction that would break everything.

    A mole is *not* skin-coloured — that is what makes it a mole — so measured
    against the raw colour mask a real lesion scores 0.00 here and the check
    would refuse every one of them. It is measured against the hole-filled skin
    region instead, where every lesion below scores 1.00.
    """
    img = _LESIONS[name]
    mask, is_a_guess = segment_or_fallback(img)
    if is_a_guess:
        pytest.skip("no outline was found; that is a different check's business")

    assert mask_on_skin(mask, skin_region(img)) > 0.9, (
        f"{name} was judged not to be on skin — this is the failure mode that "
        "refuses every pigmented lesion"
    )


def test_the_raw_colour_mask_would_have_refused_a_real_mole() -> None:
    """Why skin_region fills holes, pinned so nobody 'simplifies' it away.

    Swapping skin_region() for skin_mask() below looks like a harmless cleanup
    and turns the gate into a device that refuses every dark lesion.
    """
    img = _lesion()
    mask, _ = segment_or_fallback(img)

    raw = mask_on_skin(mask, skin_mask(img))
    filled = mask_on_skin(mask, skin_region(img))

    assert raw < 0.5, "the fixture no longer reproduces the trap this guards"
    assert filled > 0.9


@pytest.mark.parametrize("sigma", [3, 8])
def test_a_softly_focused_lesion_still_has_a_lesion_edge(sigma) -> None:
    """Edge width has to survive a soft photo, or it refuses every kiosk capture.

    Blurring is the obvious way to make a sharp edge look like a shadow, so this
    is the check's worst realistic case. Measured: a mole blurred at sigma=8
    still scores 1.68 against a ceiling of 4.0.
    """
    img = cv2.GaussianBlur(_lesion(), (0, 0), sigma)
    mask, _ = segment_or_fallback(img)

    from services.lesion_gate import _MAX_EDGE_WIDTH

    width = edge_width(img, mask)
    assert width < _MAX_EDGE_WIDTH / 2.0, (
        f"a lesion blurred at sigma={sigma} measured {width:.2f} against a ceiling of "
        f"{_MAX_EDGE_WIDTH} — raise SKIN_GATE_MAX_EDGE_WIDTH"
    )


@pytest.mark.parametrize("name", sorted(_LESIONS))
def test_a_real_lesion_stands_out_from_the_skin_around_it(name) -> None:
    """The z check, asserted on the measure rather than on the refusal code.

    It has to be pinned this way. ``contrast_z`` reuses the ``plain_skin`` code,
    which predates it, so a test that only looked at codes could not tell a
    lesion refused by the new check from one refused by the old tone-spread
    test — and would quietly pass while the new check turned lesions away.
    """
    from services.lesion_gate import _MIN_CONTRAST_Z, skin_region, spot_signals

    img = _LESIONS[name]
    mask, is_a_guess = segment_or_fallback(img)
    if is_a_guess:
        pytest.skip("no outline was found; that is a different check's business")

    z = spot_signals(img, mask, skin_geometry=skin_region(img))["z"]
    assert z >= _MIN_CONTRAST_Z * 2.0, (
        f"{name} scored z={z:.2f} against a floor of {_MIN_CONTRAST_Z} — less "
        "than 2x margin; lower SKIN_GATE_MIN_CONTRAST_Z"
    )


def test_a_mole_photographed_under_a_hard_shadow_is_refused_not_measured() -> None:
    """A known, deliberate cost of the soft-edge check, pinned so it stays known.

    This frame holds a real mole. It is refused, and that is the intended
    behaviour rather than a bug being tolerated: measured, the segmenter
    outlines the *shadow* and not the mole — 25% of the frame against the mole's
    8.7% — so the scan that used to "succeed" here was reporting the asymmetry,
    border and diameter of a patch of shadow, and calling it benign.

    "That looks like a shadow — even out the light, then put the ring over the
    mole" is the honest answer, and it is overridable for anyone who disagrees.

    If this test ever starts failing because the frame now passes, check *what
    got outlined* before celebrating.
    """
    result, _ = _scan(_lesion(_shadow(_skin())))

    assert result["blocked"] is True
    assert result["frame_check"].code == "soft_edge"
    assert result["frame_check"].can_override is True


@pytest.mark.parametrize("name", sorted(_LESIONS))
def test_no_lesion_is_refused_by_one_of_the_three_new_checks(name) -> None:
    """The narrow claim, and the only one these synthetic frames can support.

    Deliberately not "every lesion gets a verdict": two of these fixtures are
    refused by checks that predate this work (a flat frame's tone spread, the
    blur gate), and pinning that here would quietly adopt someone else's
    behaviour as this file's promise. What must hold is that the three checks
    added for bare skin do not turn any of them away.
    """
    result, _ = _scan(_LESIONS[name])
    code = getattr(result.get("frame_check"), "code", "")

    assert code not in ("off_skin", "soft_edge"), (
        f"{name} was refused as {code} by a check meant for bare skin"
    )
