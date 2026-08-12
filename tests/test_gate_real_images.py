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
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.lesion_gate import has_skin, skin_fraction
from services.quality import check_quality

ROOT = Path(__file__).resolve().parent.parent
_PATTERN = str(ROOT / "datasets" / "ham10000" / "HAM10000_images_part_*" / "*.jpg")

# Ceilings, not targets. They sit well above the rates measured on 2026-08-13
# (hard-quality 1.0%, no-skin 2.0%) so ordinary jitter does not fail the build,
# but far below the pre-fix rates (72.5% / 11.0%) that made this file necessary.
_SAMPLE_N = 60
_MAX_HARD_QUALITY_REJECT = 0.08
_MAX_NO_SKIN_REJECT = 0.10


def _sample_paths() -> list[str]:
    files = sorted(glob.glob(_PATTERN))
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
