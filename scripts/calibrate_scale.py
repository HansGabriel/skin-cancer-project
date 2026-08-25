#!/usr/bin/env python3
"""Turn a photograph of a ruler into the device's real pixels-per-millimetre.

Why this exists
---------------
``pixels_per_mm`` has always defaulted to ``10.0``, and that number came from
nowhere — no lens, no working distance, no measurement. Everything downstream
inherited it: the "About N mm across" line on the results screen, the ABCDE
"D" tier that fires at 6mm, and now the size check in ``services.lesion_gate``.
A photograph of a face once reported "About 46 mm across" on that basis.

Once the housing fixes the working distance, the field of view is a constant
and the scale becomes a real measurement taken once.

How to use it
-------------
1. Refocus the camera for the working distance the housing will hold, then put
   a ruler where the skin will sit and take a photo through the app
   ("New check" -> "Take the photo"). Any saved capture works.
2. Read off how many millimetres span the full width of that photo.
3. Run:

       python3 scripts/calibrate_scale.py ruler.jpg --span-mm 21.5

4. Put the printed line in launch_kiosk.sh, next to the other exports, and
   restart the kiosk.

The captured frame is a centre square crop of the sensor
(``services/pi_camera.py``), so "across the full width" means across that
square, not across anything you can see in the live preview outside the guide.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

# Kept in step with services.lesion_gate._MAX_LESION_MM, including its override,
# so that setting the env var does not make this script print a number the gate
# will not honour. Duplicated rather than imported so the script has no
# dependency on the frontend package; tests/test_not_a_lesion.py pins the two
# defaults together, the way tests/test_kiosk_exit.py pins the quit-flag path.
_MAX_LESION_MM_DEFAULT = 20.0


def _max_lesion_mm() -> float:
    try:
        return float(os.environ.get("SKIN_GATE_MAX_LESION_MM", "") or _MAX_LESION_MM_DEFAULT)
    except ValueError:
        return _MAX_LESION_MM_DEFAULT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="a photo of a ruler taken through the device")
    parser.add_argument(
        "--span-mm",
        type=float,
        required=True,
        help="millimetres visible across the full width of that photo",
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"No such file: {args.image}", file=sys.stderr)
        return 1
    if args.span_mm <= 0:
        print("--span-mm must be greater than zero.", file=sys.stderr)
        return 1

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Could not read {args.image} as an image.", file=sys.stderr)
        return 1

    height, width = img.shape[:2]
    pixels_per_mm = width / args.span_mm

    print(f"image            : {width} x {height}")
    print(f"span across width: {args.span_mm:.2f} mm")
    print(f"pixels per mm    : {pixels_per_mm:.2f}")
    print()
    print(f"field of view    : {args.span_mm:.1f} mm x {height / pixels_per_mm:.1f} mm")
    max_mm = _max_lesion_mm()
    print(
        f"size check       : refuses anything wider than {max_mm:.0f} mm, "
        f"i.e. {max_mm * pixels_per_mm:.0f} px in this frame"
    )
    if args.span_mm > 60:
        print()
        print(
            "  Note: a field of view this wide is not a close-up. If this is the\n"
            "  housing's working distance, a lesion will occupy very few pixels and\n"
            "  the size check will never fire. Check the lens is focused for ~6cm."
        )
    print()
    print("Add to launch_kiosk.sh (and run_pi.sh if you test there):")
    print()
    print(f"    export SKIN_PIXELS_PER_MM={pixels_per_mm:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
