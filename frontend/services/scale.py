"""How many pixels a millimetre of skin covers — and whether anyone measured it.

``pixels_per_mm`` has always defaulted to ``10.0``, and that number was never
derived from a lens, a sensor or a working distance. It is why the results
screen could report "About 46 mm across" for a photograph of somebody's face.
Everything downstream inherits it: the diameter shown to the user, the ABCDE
"D" tier that fires at 6mm, and the size check in ``services.lesion_gate``.

Two different questions get asked of that number, and conflating them is a bug:

* **What should we display?** The placeholder is fine here. A wrong millimetre
  figure next to a real photo is a cosmetic problem.
* **May we refuse a scan because of it?** Only if it was actually measured, and
  only for a capture taken through the optics it was measured for. A refusal on
  the strength of a made-up constant is inventing a reason.

The second question is what ``trusted_pixels_per_mm`` answers, and it
deliberately ignores the staff override in Settings. Staff can retune the
displayed scale; they cannot thereby arm a hard, non-overridable refusal with a
number nobody took off a ruler.
"""

from __future__ import annotations

import os

# The historical placeholder. Kept as a named constant so the three places that
# used to spell "10.0" cannot drift apart again.
DEFAULT_PIXELS_PER_MM = 10.0


def measured_pixels_per_mm() -> float | None:
    """The scale calibrated off a ruler, or ``None`` if nobody has measured it.

    Set ``SKIN_PIXELS_PER_MM`` from ``scripts/calibrate_scale.py`` once the
    housing fixes the working distance. A malformed or non-positive value is
    treated as unset rather than raising: this is read on the scan path, on a
    device with no keyboard.
    """
    raw = os.environ.get("SKIN_PIXELS_PER_MM")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def default_pixels_per_mm() -> float:
    """What to seed the UI with: the measured scale if there is one."""
    return measured_pixels_per_mm() or DEFAULT_PIXELS_PER_MM


def trusted_pixels_per_mm(*, from_device_camera: bool) -> float | None:
    """Scale the size check may refuse a photo on, or ``None`` for "do not".

    Both conditions are required. An uploaded photo — every web-app scan, and
    "Choose a picture" on the device — was taken at an unknown distance through
    unknown optics, so no scale applies to it however well the device itself is
    calibrated.
    """
    return measured_pixels_per_mm() if from_device_camera else None
