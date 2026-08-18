"""Does the answer change when the photo does, but the lesion does not?

This is the gap the rest of the suite left open. Every other model-touching
test (test_decision_parity.py, test_temperature.py) feeds *synthetic
probability vectors* — no test ever ran an image through the TFLite graph. So
"the scanner is accurate across angles and lighting" was an assumption with
nothing measuring it.

Two levels here, because only one of them can run without the dataset:

1. **Dihedral invariance is exact, by construction.** The TTA view set is
   ``{identity, hflip, vflip, rot180}``, and that set is *closed* under each of
   those three transforms. Averaging over a closed set gives the same average
   whichever member you start from, so flipping the input cannot change the
   output at all — not "by a small margin", exactly. That is a property of the
   code and is tested here directly.

2. **Everything else is empirical** — 30 degrees, 45 degrees, and exposure
   changes are not in the TTA set, so nothing guarantees them and they have to
   be measured on real dermoscopy. Skipped without ``datasets/ham10000``, the
   same way tests/test_gate_real_images.py is.

``samples/*.jpg`` are deliberately NOT used: ``samples/README.md`` documents
them as synthetic colour patches, and measured against the deployed model all
three saturate at p(cancer) = 0.000. An invariance test on them passes for the
wrong reason. That is the same trap that once calibrated the focus threshold
against those files and refused 72% of genuine lesions.
"""

from __future__ import annotations

import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.tflite_shared import _flip_rgb, decide_index, run_inference_on_rgb

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "skin_classifier.tflite"
_PATTERN = str(ROOT / "datasets" / "ham10000" / "HAM10000_images_part_*" / "*.jpg")

# The four views backend.tflite_shared averages over.
TTA_MODES = ("identity", "hflip", "vflip", "rot180")

# Ceilings, not targets — the same convention as test_gate_real_images.py. A
# rotation or a lighting change must not move the screening decision; the
# probability is allowed to drift a little underneath it.
_SAMPLE_N = 12
_MAX_DECISION_FLIPS = 0.0
_MAX_P_CANCER_DRIFT = 0.20


def _apply(image: np.ndarray, mode: str) -> np.ndarray:
    return image if mode == "identity" else _flip_rgb(image, mode)


# --- Level 1: always runs, no model and no dataset needed -----------------


@pytest.mark.parametrize("transform", ["hflip", "vflip", "rot180"])
def test_tta_view_set_is_closed_under_its_own_transforms(transform: str) -> None:
    """Why TTA makes the scanner exactly flip-invariant.

    Applying any of these to the input permutes the four views among
    themselves. The average is order-independent, so the averaged probabilities
    are identical — this is the guarantee, and it is what would break if
    someone ever added a fifth view (say a 90-degree rotation) without adding
    its partners.
    """
    rng = np.random.default_rng(11)
    image = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)

    got = {_apply(_apply(image, transform), mode).tobytes() for mode in TTA_MODES}
    want = {_apply(image, mode).tobytes() for mode in TTA_MODES}
    assert got == want, f"applying {transform} left the TTA view set"


def test_tta_modes_are_the_ones_this_guarantee_assumes() -> None:
    """Pins the set itself, so the closure test above cannot silently pass on
    a view list that no longer matches what run_inference_on_rgb uses."""
    import inspect

    src = inspect.getsource(run_inference_on_rgb)
    assert '("identity", "hflip", "vflip", "rot180")' in src


# --- Level 2: real dermoscopy, skipped without the dataset ----------------


def _sample_paths() -> list[str]:
    files = sorted(glob.glob(_PATTERN))
    if not files:
        return []
    return random.Random(42).sample(files, min(_SAMPLE_N, len(files)))


_PATHS = _sample_paths()
pytestmark_reason = "HAM10000 not present in datasets/ (or model missing)"


@pytest.fixture(scope="module")
def interpreter():
    if not MODEL.is_file():
        pytest.skip("models/skin_classifier.tflite not present")
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(MODEL), num_threads=4)
    interp.allocate_tensors()
    return interp


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def _expose(image: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _decide(interp, rgb: np.ndarray) -> tuple[int, float]:
    probs, _ms = run_inference_on_rgb(rgb, interp, use_tta=True)
    return decide_index(probs), float(probs[1] + probs[2])


@pytest.mark.skipif(not _PATHS, reason=pytestmark_reason)
@pytest.mark.parametrize("degrees", [30, 45, 60])
def test_off_axis_rotation_does_not_flip_the_decision(interpreter, degrees: int) -> None:
    """Nobody holds a dermatoscope at a repeatable angle.

    These rotations are outside the TTA set, so nothing in the code guarantees
    them — the training augmentation (RandomRotation 0.2 = +/-72 degrees) is
    what is being measured here.
    """
    flips = 0
    drifts = []
    for path in _PATHS:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        base_idx, base_p = _decide(interpreter, image)
        rot_idx, rot_p = _decide(interpreter, _rotate(image, degrees))
        flips += int(base_idx != rot_idx)
        drifts.append(abs(rot_p - base_p))
    rate = flips / float(len(_PATHS))
    assert rate <= _MAX_DECISION_FLIPS, (
        f"{degrees} deg flipped the decision on {flips}/{len(_PATHS)} images"
    )
    assert max(drifts) <= _MAX_P_CANCER_DRIFT, f"p(cancer) drifted {max(drifts):.3f}"


@pytest.mark.skipif(not _PATHS, reason=pytestmark_reason)
@pytest.mark.parametrize("gain", [0.7, 1.3])
def test_lighting_change_does_not_flip_the_decision(interpreter, gain: float) -> None:
    """A hall lit by a window in the morning and by fluorescents at 4pm."""
    flips = 0
    for path in _PATHS:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        base_idx, _ = _decide(interpreter, image)
        lit_idx, _ = _decide(interpreter, _expose(image, gain))
        flips += int(base_idx != lit_idx)
    rate = flips / float(len(_PATHS))
    assert rate <= _MAX_DECISION_FLIPS, (
        f"exposure x{gain} flipped the decision on {flips}/{len(_PATHS)} images"
    )
