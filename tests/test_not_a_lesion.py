"""Photos that are not one spot on skin must not get a verdict.

The gap this closes was measured, not theorised. A real portrait passed every
stage of the gate and came back "NOTHING STOOD OUT": skin fraction 0.13 against
a floor of 0.08, tone spread 99.6 against 12, a segmented "lesion" covering
0.176 of the frame against an ideal lesion size of 0.18, and a Lab contrast of
63 against a threshold of 5. Every check answered yes, because every check was
asking "is this skin, and is there a dark blob on it" — and a face is skin with
eyes, nostrils and a mouth on it.

The classifier has three labels: benign, pre-cancerous, malignant. There is no
"not a lesion" among them, so anything reaching it is called one of the three.
That is why these refusals are hard.

**The fixtures are generated, not photographed.** What made the face pass is the
*distribution of dark area* — several regions sharing it rather than one holding
it — and a skin-toned field with several blobs on it reproduces that exactly.
No licensing, no faces of real people in the repository, and deterministic in
CI. Verified against a real portrait during development; the synthetic frames
score the same way (dominance 0.58 vs 0.50).

Half of these tests are the other direction, and they matter more: an unusual
real lesion refused with no way through is the failure this project cannot
afford.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from services.lesion_gate import (
    check_skin,
    quick_reject,
    check_scale,
    check_screen,
    check_structure,
)

_SEED = 20260825


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _skin(h: int = 900, w: int = 900, tone: tuple[int, int, int] = (196, 158, 132)) -> np.ndarray:
    """Skin-toned field with grain, so focus and tone-spread checks behave."""
    rng = _rng()
    base = np.full((h, w, 3), tone, np.uint8)
    mottle = rng.integers(-14, 14, (h, w, 1)).repeat(3, axis=2)
    grain = rng.integers(-9, 9, (h, w, 3))
    return np.clip(base.astype(np.int16) + mottle + grain, 0, 255).astype(np.uint8)


def _one_lesion() -> np.ndarray:
    img = _skin()
    cv2.circle(img, (450, 450), 150, (78, 62, 74), -1)
    return img


def _multi_lobed() -> np.ndarray:
    """An irregular melanoma: one lesion, two lobes, asymmetric."""
    img = _skin()
    cv2.ellipse(img, (450, 450), (190, 110), 25, 0, 360, (74, 58, 70), -1)
    cv2.ellipse(img, (520, 400), (90, 70), -10, 0, 360, (52, 40, 52), -1)
    return img


def _lesion_with_satellite() -> np.ndarray:
    img = _skin()
    cv2.circle(img, (430, 440), 140, (76, 60, 72), -1)
    cv2.circle(img, (650, 300), 42, (70, 55, 66), -1)
    return img


def _lesion_under_hair() -> np.ndarray:
    rng = _rng()
    img = _skin()
    cv2.circle(img, (430, 440), 140, (76, 60, 72), -1)
    for _ in range(60):
        x, y = (int(v) for v in rng.integers(0, 900, 2))
        cv2.line(img, (x, y), (x + int(rng.integers(-200, 200)), y + int(rng.integers(-200, 200))),
                 (38, 30, 28), 3)
    return img


def _lesion_dark_skin() -> np.ndarray:
    img = _skin(tone=(120, 86, 66))
    cv2.circle(img, (450, 450), 150, (48, 34, 30), -1)
    return img


def _pale_lesion() -> np.ndarray:
    """Amelanotic: barely darker than the skin around it, so almost no structure."""
    img = _skin()
    cv2.circle(img, (450, 450), 150, (214, 150, 150), -1)
    return img


def _face_like() -> np.ndarray:
    """Several separate dark marks — the shape a face presents to this gate."""
    img = _skin()
    for x, y, r in ((330, 350, 62), (570, 350, 62), (450, 520, 40), (450, 660, 95), (450, 250, 30)):
        cv2.circle(img, (x, y), r, (58, 44, 52), -1)
    return img


def _screen(step: int = 4) -> np.ndarray:
    img = _one_lesion()
    img[::step, :] = (60, 50, 45)
    img[:, ::step] = (60, 50, 45)
    return img


def _disc_mask(radius_px: int) -> np.ndarray:
    mask = np.zeros((900, 900), np.uint8)
    cv2.circle(mask, (450, 450), radius_px, 255, -1)
    return mask


# --------------------------------------------------------------- must refuse

@pytest.mark.parametrize("frame", [_face_like(), _face_like()[0:600, 150:750]])
def test_several_dark_marks_are_refused(frame) -> None:
    check = check_structure(frame)
    assert check is not None, "a frame of several separate marks reached the classifier"
    assert check.is_lesion_photo is False
    assert check.code == "structure"


def test_that_refusal_offers_no_way_through() -> None:
    """Forcing a face through cannot produce a cautious answer, only a wrong one."""
    check = check_structure(_face_like())
    assert check is not None
    assert check.severity == "hard"
    assert check.can_override is False


def test_a_photo_of_a_screen_is_refused() -> None:
    check = check_screen(_screen())
    assert check is not None
    assert check.code == "screen"


def test_the_screen_refusal_offers_no_way_through_either() -> None:
    """Hard, because the threshold has the margin to justify it.

    The worst any photograph measured was a dermatoscope reticle at 136 and the
    line is at 400 — 2.9x clear. Coarse grids below the line are missed rather
    than caught softly; that is the trade, and the semantic stage is the backstop.
    """
    check = check_screen(_screen())
    assert check is not None
    assert check.severity == "hard"
    assert check.can_override is False


def test_something_far_too_large_to_be_a_spot_is_refused() -> None:
    # 300px across at 10 px/mm is 30mm — wider than any skin lesion.
    check = check_scale(_disc_mask(150), pixels_per_mm=10.0)
    assert check is not None
    assert check.code == "too_large"
    assert check.can_override is False


# ---------------------------------------------------------------- must pass

@pytest.mark.parametrize(
    "name,frame",
    [
        ("single lesion", _one_lesion()),
        ("multi-lobed melanoma", _multi_lobed()),
        ("lesion with a satellite", _lesion_with_satellite()),
        ("lesion under heavy hair", _lesion_under_hair()),
        ("lesion on dark skin", _lesion_dark_skin()),
    ],
)
def test_real_lesions_are_never_refused_as_scenes(name, frame) -> None:
    assert check_structure(frame) is None, f"{name} was refused"
    assert check_screen(frame) is None, f"{name} was called a screen"


def test_a_frame_with_no_real_structure_is_not_called_a_scene() -> None:
    """Bare skin and a barely-visible lesion threshold into a scatter of noise.

    Without the contrast floor both look like "several dark marks" and would be
    refused as scenes — a pale lesion being exactly the kind of thing that must
    not be turned away.
    """
    assert check_structure(_skin()) is None
    assert check_structure(_pale_lesion()) is None


def test_size_is_not_judged_when_the_scale_was_never_measured() -> None:
    """Every web-app upload lands here: no optics, no distance, no scale.

    ``pixels_per_mm`` defaults to a placeholder of 10.0 that nobody derived from
    a lens, so refusing an upload on the strength of it would be inventing a
    reason.
    """
    assert check_scale(_disc_mask(150), pixels_per_mm=None) is None
    assert check_scale(_disc_mask(150), pixels_per_mm=0.0) is None
    assert check_scale(None, pixels_per_mm=10.0) is None


def test_a_normal_sized_spot_passes_the_size_check() -> None:
    # 100px across at 10 px/mm is 10mm.
    assert check_scale(_disc_mask(50), pixels_per_mm=10.0) is None


def test_a_blurred_lesion_is_refused_softly_and_can_be_forced() -> None:
    """The refusals that judge the *photo* must always leave a way through.

    A soft lesion is the commonest reason a genuine spot gets turned away, and
    the person holding it is the one who can see it is real. Only refusals about
    the *subject* — this is not one spot, this is a screen, this is too large,
    this is not something the model can read — are hard.
    """
    from services.pipeline import run_pipeline

    blurred = cv2.GaussianBlur(_one_lesion(), (0, 0), 12)
    result = run_pipeline(_CountingBackend(), _jpeg(blurred),
                          pixels_per_mm=10.0, strict_quality=False)

    assert result["blocked"] is True
    assert result["verdict"].headline == "TAKE ANOTHER PHOTO"
    # No frame check at all: this is a quality refusal, and results_view defaults
    # an absent check to overridable. Pinned because "hard by default" would
    # silently turn every quality refusal into a dead end.
    check = result.get("frame_check")
    assert getattr(check, "can_override", True) is True


# ------------------------------------------------------------------ wording

def test_the_refusals_say_what_is_wrong_without_jargon() -> None:
    banned = ("mask", "segmentation", "HSV", "YCrCb", "Lab", "threshold", "fraction", "pixel",
              "blob", "Otsu", "frequency", "Mahalanobis")
    checks = [check_structure(_face_like()), check_screen(_screen()),
              check_scale(_disc_mask(150), pixels_per_mm=10.0)]
    for check in checks:
        assert check is not None
        for sentence in check.reasons:
            assert sentence[0].isupper(), sentence
            assert sentence.endswith("."), sentence
            for word in banned:
                assert word.lower() not in sentence.lower(), (word, sentence)


# ----------------------------------------------------------------- the path

def test_a_scene_is_refused_by_the_cheap_pre_pass() -> None:
    """It must be caught before any segmentation work is paid for."""
    _no_skin, skin = check_skin(_face_like())
    early = quick_reject(_face_like(), skin=skin)
    assert early is not None
    assert early.code == "structure"


def test_a_real_lesion_survives_the_cheap_pre_pass() -> None:
    _no_skin, skin = check_skin(_one_lesion())
    assert quick_reject(_one_lesion(), skin=skin) is None


# --------------------------------------------------------- through the whole pipeline

def _jpeg(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


class _CountingBackend:
    """Confidently benign, so anything that reaches it is visible in the result."""

    backend_id = "mock"

    def __init__(self) -> None:
        self.calls = 0

    def scan(self, image_jpg_bytes=None):
        from backend.contracts import ScanResult

        self.calls += 1
        return ScanResult(
            label="benign",
            confidence=95.0,
            probs={"benign": 95.0, "pre_cancerous": 3.0, "malignant": 2.0},
        )


def test_a_face_never_reaches_the_classifier() -> None:
    """The measured bug: this returned "NOTHING STOOD OUT" for a portrait."""
    from services.pipeline import run_pipeline

    backend = _CountingBackend()
    result = run_pipeline(backend, _jpeg(_face_like()), pixels_per_mm=10.0, strict_quality=False)

    assert result["blocked"] is True
    assert backend.calls == 0, "the model was asked to classify a scene"
    assert result["scan_result"] is None


def test_the_refusal_says_what_is_actually_wrong() -> None:
    """"NO SKIN SPOT FOUND" is right for a wall and nonsense for a face."""
    from services.pipeline import run_pipeline

    result = run_pipeline(_CountingBackend(), _jpeg(_face_like()),
                          pixels_per_mm=10.0, strict_quality=False)

    assert result["verdict"].headline == "THIS IS NOT ONE SPOT"
    assert result["frame_check"].can_override is False


def test_refusing_a_scene_costs_nothing(caplog) -> None:
    """It must exit before enhancement and segmentation, which dominate a scan."""
    from services.pipeline import run_pipeline

    result = run_pipeline(_CountingBackend(), _jpeg(_face_like()),
                          pixels_per_mm=10.0, strict_quality=False)

    stages = result.get("stage_ms") or {}
    for expensive in ("enhance", "segment", "model", "abcde"):
        assert expensive not in stages, f"a refused scene paid for {expensive}"


@pytest.mark.parametrize(
    "name,frame",
    [
        ("single lesion", _one_lesion()),
        ("multi-lobed melanoma", _multi_lobed()),
        ("lesion on dark skin", _lesion_dark_skin()),
        ("lesion under heavy hair", _lesion_under_hair()),
    ],
)
def test_real_lesions_still_reach_the_classifier(name, frame) -> None:
    from services.pipeline import run_pipeline

    backend = _CountingBackend()
    result = run_pipeline(backend, _jpeg(frame), pixels_per_mm=10.0, strict_quality=False)

    assert not result.get("blocked"), f"{name} was refused: {result['verdict'].headline}"
    assert backend.calls == 1, f"{name} never reached the model"


def test_an_upload_is_never_refused_for_its_size() -> None:
    """The web app's whole input is uploads, and none of them carry a scale."""
    from services.pipeline import run_pipeline

    backend = _CountingBackend()
    result = run_pipeline(backend, _jpeg(_one_lesion()), pixels_per_mm=10.0,
                          strict_quality=False)  # trusted_pixels_per_mm defaults to None

    assert not result.get("blocked")
    assert backend.calls == 1


def test_the_device_camera_can_refuse_something_too_large() -> None:
    """Only the fixed-optics capture path passes a scale the gate may act on.

    At 40 px/mm a 300px-wide spot measures 7.5mm and passes; the same frame at
    4 px/mm measures 75mm, which is a face, not a mole.
    """
    from services.pipeline import run_pipeline

    frame = _jpeg(_one_lesion())

    ok = run_pipeline(_CountingBackend(), frame, pixels_per_mm=40.0,
                      strict_quality=False, trusted_pixels_per_mm=40.0)
    assert not ok.get("blocked")

    too_big = run_pipeline(_CountingBackend(), frame, pixels_per_mm=4.0,
                           strict_quality=False, trusted_pixels_per_mm=4.0)
    assert too_big["blocked"] is True
    assert too_big["frame_check"].code == "too_large"
    assert too_big["verdict"].headline == "TOO BIG TO BE A SPOT"


# ------------------------------------------- the outline the scanner invented

def test_an_invented_outline_is_reported_as_one() -> None:
    """``segment_or_fallback`` must say when it drew a circle instead of finding one.

    When no candidate is plausible the segmenter returns a plain disc in the
    middle of the frame so ABCDE cannot crash. Everything measured from it —
    asymmetry, border, diameter — then describes that disc rather than anything
    in the photo. ``segment_safe`` threw the distinction away, and the cost was
    exact: ``check_spot`` has a "No spot could be picked out" branch that was
    **unreachable through the pipeline**, because it was only ever handed a mask
    that existed.
    """
    from services.segmentation import segment_or_fallback

    flat = np.full((256, 256, 3), 128, np.uint8)
    _mask, is_a_guess = segment_or_fallback(flat, use_grabcut=False)
    assert is_a_guess is True

    _mask, is_a_guess = segment_or_fallback(
        cv2.resize(_one_lesion(), (256, 256)), use_grabcut=False
    )
    assert is_a_guess is False, "a real lesion's outline was reported as invented"


def test_the_no_spot_refusal_is_reachable() -> None:
    """The branch the flag exists to make reachable."""
    from services.lesion_gate import check_spot

    flat = np.full((256, 256, 3), 128, np.uint8)
    disc = np.zeros((256, 256), np.uint8)
    cv2.circle(disc, (128, 128), 80, 255, -1)

    check = check_spot(flat, disc, skin=1.0, mask_is_a_guess=True)
    assert check.is_lesion_photo is False
    assert check.code == "no_spot"
    assert check.reasons == ("No spot could be picked out on the skin.",)


def test_a_real_outline_is_still_measured() -> None:
    from services.lesion_gate import check_spot
    from services.segmentation import segment_or_fallback

    frame = _one_lesion()
    mask, is_a_guess = segment_or_fallback(frame, use_grabcut=False)
    check = check_spot(frame, mask, skin=1.0, mask_is_a_guess=is_a_guess)
    assert check.is_lesion_photo is True


# --------------------------------------------------- constants that must agree

def test_the_calibration_script_agrees_with_the_gate() -> None:
    """``scripts/calibrate_scale.py`` prints the size the gate will enforce.

    The threshold is duplicated there so the script does not import the frontend
    package. Same shape as ``tests/test_kiosk_exit.py`` pinning the quit-flag
    path against ``launch_kiosk.sh``: duplication is fine, silent drift is not.
    """
    import re

    from services.lesion_gate import _MAX_LESION_MM

    source = (Path(__file__).resolve().parents[1] / "scripts" / "calibrate_scale.py").read_text()
    match = re.search(r"_MAX_LESION_MM_DEFAULT = ([0-9.]+)", source)
    assert match, "the script no longer declares a default to compare against"
    assert float(match.group(1)) == _MAX_LESION_MM
    # And it must read the same override, or the printed number is a lie.
    assert "SKIN_GATE_MAX_LESION_MM" in source
