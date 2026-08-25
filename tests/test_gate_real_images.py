"""The gate must let real lesions through — measured on real dermoscopy.

Every other gate test asserts that *junk* is refused, using synthetic frames.
Nothing asserted the other direction, and that is how the shipped thresholds
came to refuse **72% of genuine HAM10000 images** while the suite stayed green:

- the focus threshold was calibrated against ``samples/``, which
  ``samples/README.md`` documents as synthetic colour patches. They score
  435-469; real dermoscopy has a median of 79. The soft threshold sat at 120.
- the skin detector's Cb ceiling of 127 excluded polarized and oil-immersion
  dermoscopy, which sits at Cb 133-138 — the images this device exists to read.

A false rejection is not a harmless "try again". It is the one failure mode the
user actually hits, and it silently converts a screening tool into a device
that refuses to look at skin.

Skipped when HAM10000 is not on disk, so CI without the dataset stays green.
The sample is a fixed seed and a fixed size so the numbers below are
reproducible, and small enough that the suite stays fast.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.lesion_gate import has_skin, skin_fraction
from services.quality import check_quality

ROOT = Path(__file__).resolve().parent.parent
# Both spellings, and an override. models/test_split.csv records the training
# machine's paths as ``ham10000_images_part_1`` in lower case while this pattern
# was upper — and glob is case-sensitive on Linux, which is where the dataset
# actually lives. The test would have skipped silently on the one machine able
# to run it. SKIN_HAM10000_DIR points somewhere else entirely if needed.
_ROOTS = [Path(d) for d in (os.environ.get("SKIN_HAM10000_DIR"),) if d]
_ROOTS += [ROOT / "datasets" / "ham10000"]
_PATTERNS = [
    str(base / part / "*.jpg")
    for base in _ROOTS
    for part in ("HAM10000_images_part_*", "ham10000_images_part_*", "*")
]

# Ceilings, not targets. They sit well above the rates measured on 2026-08-13
# (hard-quality 1.0%, no-skin 2.0%) so ordinary jitter does not fail the build,
# but far below the pre-fix rates (72.5% / 11.0%) that made this file necessary.
_SAMPLE_N = 60
_MAX_HARD_QUALITY_REJECT = 0.08
_MAX_NO_SKIN_REJECT = 0.10


def _sample_paths() -> list[str]:
    files: list[str] = []
    for pattern in _PATTERNS:
        files = sorted(glob.glob(pattern))
        if files:
            break
    if not files:
        return []
    return random.Random(42).sample(files, min(_SAMPLE_N, len(files)))


_PATHS = _sample_paths()
pytestmark = pytest.mark.skipif(not _PATHS, reason="HAM10000 not present in datasets/")


def _load(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def test_real_dermoscopy_is_not_hard_rejected_as_unreadable() -> None:
    """``ok=False`` is the hard stop — it blocks regardless of strict mode."""
    rejected = [p for p in _PATHS if not check_quality(_load(p))["ok"]]
    rate = len(rejected) / len(_PATHS)
    assert rate <= _MAX_HARD_QUALITY_REJECT, (
        f"{rate:.1%} of real dermoscopic images hard-rejected as unreadable "
        f"(limit {_MAX_HARD_QUALITY_REJECT:.0%}). Check _BLUR_HARD in "
        f"services/quality.py. Examples: {[Path(p).name for p in rejected[:5]]}"
    )


def test_real_dermoscopy_is_recognised_as_skin() -> None:
    """The stage that produced "No skin was found in this photo."."""
    rejected = [p for p in _PATHS if not has_skin(_load(p))]
    rate = len(rejected) / len(_PATHS)
    assert rate <= _MAX_NO_SKIN_REJECT, (
        f"{rate:.1%} of real dermoscopic images read as 'no skin' "
        f"(limit {_MAX_NO_SKIN_REJECT:.0%}). Check _SKIN_MIN and the Cb ceiling "
        f"in services/lesion_gate.py. Examples: {[Path(p).name for p in rejected[:5]]}"
    )


def test_most_real_dermoscopy_clears_the_soft_focus_threshold() -> None:
    """The advisory that blocks whenever strict mode is on.

    This is the specific number that made the app unusable: at the old
    threshold of 120 this assertion would have failed at 72.5%.
    """
    soft = [p for p in _PATHS if "blur" in {c for c, _l, _s in check_quality(_load(p))["reason_details"]}]
    rate = len(soft) / len(_PATHS)
    assert rate <= 0.25, (
        f"{rate:.1%} of real dermoscopic images flagged as soft/blurry "
        "(limit 25%). Check _BLUR_SOFT in services/quality.py — every one of "
        "these is refused outright when 'Refuse photos that are not very "
        "clear' is on."
    )


def test_skin_fraction_separates_real_skin_from_neutral_surfaces() -> None:
    """The margin the _SKIN_MIN threshold has to sit inside.

    Guards the trade directly: the median real image must stay far above the
    threshold, so tightening it later cannot quietly re-break real photos.
    """
    from services.lesion_gate import _SKIN_MIN

    fractions = np.array([skin_fraction(_load(p)) for p in _PATHS])
    median = float(np.median(fractions))
    assert median > _SKIN_MIN * 3, (
        f"median real skin_fraction {median:.3f} is not comfortably above "
        f"_SKIN_MIN {_SKIN_MIN:.3f} — the gate has no margin left"
    )


# --------------------------------------------------------------------------
# The checks added to stop a face being read as a lesion. Both refuse a photo
# outright, and the structure one refuses it with no way through — so the
# false-rejection rate on real dermoscopy is the number that matters, and it is
# the number that cannot be measured on the machine they were written on.
#
# They were calibrated against synthetic lesion variants (single, multi-lobed,
# satellite, heavy hair, dark skin, pale) plus one real photograph for the
# reject side. This file is where that calibration meets real lesions.
# --------------------------------------------------------------------------

# Ceilings, not targets. A real lesion is one region on skin, so the structure
# check should essentially never fire on this dataset; anything above a couple
# of percent means the dominance threshold is wrong for real dermoscopy.
_MAX_STRUCTURE_REJECT = 0.02
_MAX_SCREEN_REJECT = 0.02


def test_real_lesions_are_not_refused_as_scenes() -> None:
    """"This photo has several dark marks in it" must be rare on real lesions."""
    from services.lesion_gate import check_structure

    refused = [p for p in _PATHS if check_structure(_load(p)) is not None]
    rate = len(refused) / len(_PATHS)
    assert rate <= _MAX_STRUCTURE_REJECT, (
        f"structure check refused {rate:.0%} of real lesions "
        f"({len(refused)}/{len(_PATHS)}); raise SKIN_GATE_STRUCTURE_DOMINANCE or "
        f"SKIN_GATE_STRUCTURE_CONTRAST. Examples: {refused[:3]}"
    )


def test_real_lesions_are_not_mistaken_for_screens() -> None:
    """Dermatoscope reticles and rulers are the plausible false positive here."""
    from services.lesion_gate import check_screen

    refused = [p for p in _PATHS if check_screen(_load(p)) is not None]
    rate = len(refused) / len(_PATHS)
    assert rate <= _MAX_SCREEN_REJECT, (
        f"screen check refused {rate:.0%} of real lesions "
        f"({len(refused)}/{len(_PATHS)}); raise SKIN_GATE_MOIRE_RATIO. "
        f"Examples: {refused[:3]}"
    )


def test_the_structure_signal_separates_lesions_from_scenes() -> None:
    """Report the margin, so a threshold change can be judged rather than guessed."""
    from services.lesion_gate import _STRUCTURE_MAX_DOMINANCE, dark_structure

    dominances = [dark_structure(_load(p))[1] for p in _PATHS]
    median = float(np.median(dominances))
    worst = float(np.min(dominances))
    assert median > _STRUCTURE_MAX_DOMINANCE, (
        f"real lesions have a median dominance of {median:.2f}, at or below the "
        f"{_STRUCTURE_MAX_DOMINANCE} threshold — the signal does not separate "
        f"them from scenes on this data (worst {worst:.2f})"
    )
