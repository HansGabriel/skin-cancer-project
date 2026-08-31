#!/usr/bin/env python3
"""Measure the content gate's signals over a folder of real photographs.

Why this exists
---------------
The three checks that stop a photograph of bare skin reaching the classifier —
``off_skin``, ``soft_edge`` and the ``plain_skin`` z-score in
``services/lesion_gate.py`` — have thresholds that were set from *generated*
frames, because no photographs of the failure existed to set them from.

This repo has already paid for that shortcut once. ``tests/test_gate_real_images.py``
records a synthetic-only calibration that refused **72% of genuine HAM10000
images** while the whole suite stayed green. So the thresholds shipped
deliberately loose, every scan logs its own measurements, and this script is how
a folder of real images turns into the numbers that replace them.

How to use it
-------------
Point it at each set separately and keep the printed tables:

    # must PASS — real lesions. Run this one first and run it on the rig that
    # holds the dataset; it is the direction that has actually burned us.
    python3 scripts/measure_gate_signals.py \\
        --images datasets/ham10000/HAM10000_images_part_1 --name ham10000

    # must PASS — real moles photographed on the kiosk itself. HAM10000 alone
    # cannot exercise the on-skin check: dermoscopy fills the frame with skin,
    # so every image scores 1.00 for free.
    python3 scripts/measure_gate_signals.py --images captures/moles --name kiosk-moles

    # must be REFUSED — bare skin through the same camera, in varied light.
    python3 scripts/measure_gate_signals.py --images captures/bare --name bare-skin

Then pick each threshold so it sits **strictly between the must-pass set's worst
case and the must-refuse set's median, with at least 2x margin from the must-pass
worst case** — the same rule ``_MOIRE_MAX_PEAK_RATIO`` was chosen under, where
400 sits 2.9x above the worst photograph of anything. A measure that cannot meet
that rule on real images does not get to block a scan: set its env var so it can
never fire and leave it logging.

``--csv`` writes one row per image for the images that fall the wrong side of a
threshold, so a disagreement can be looked at rather than argued about.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frontend"))

_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_COLUMNS = ("skin", "on_skin", "edge_width", "contrast", "sigma_skin", "z")
_PERCENTILES = (0, 1, 5, 50, 95, 99, 100)


def _measure(path: Path) -> dict[str, float] | None:
    """Every gate signal for one image, or ``None`` if it could not be read.

    Deliberately routed through ``decode_image_bytes_to_rgb`` and the pipeline's
    own working-resolution cap rather than ``cv2.imread``: JPEG chroma
    quantisation moves the skin test and JPEG grain moves the noise floor the
    screen test divides by, so a number measured off a raw array is not the
    number the device will see. Both of this repo's threshold regressions were
    measured that way.
    """
    from backend.tflite_shared import decode_image_bytes_to_rgb
    from services.lesion_gate import check_skin, skin_region, spot_signals
    from services.pipeline import cap_working_resolution
    from services.segmentation import segment_or_fallback

    try:
        rgb = decode_image_bytes_to_rgb(path.read_bytes())
    except Exception as exc:  # noqa: BLE001 — one unreadable file must not end the run
        print(f"  skipped {path.name}: {exc}", file=sys.stderr)
        return None

    rgb = cap_working_resolution(rgb)
    _no_skin, skin = check_skin(rgb)
    geometry = skin_region(rgb)
    mask, is_a_guess = segment_or_fallback(rgb)
    if is_a_guess:
        # A centre circle nobody found in the photo. Measuring it would compare
        # skin against skin and quietly pull every percentile toward "refuse".
        return None

    row = spot_signals(rgb, mask, skin_geometry=geometry)
    row["skin"] = skin
    return row


def _table(name: str, rows: list[dict[str, float]]) -> None:
    print(f"\n=== {name} — {len(rows)} images ===")
    header = "  " + "".join(f"{c:>12s}" for c in _COLUMNS)
    print(header)
    for p in _PERCENTILES:
        values = "".join(
            f"{np.percentile([r[c] for r in rows], p):12.2f}" for c in _COLUMNS
        )
        print(f"p{p:<3d}" + values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of photographs")
    ap.add_argument("--name", required=True, help="what this set is, e.g. bare-skin")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    ap.add_argument("--csv", help="write one row per image here")
    args = ap.parse_args()

    directory = Path(args.images).expanduser()
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in _SUFFIXES)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no images under {directory}", file=sys.stderr)
        return 2

    rows: list[dict[str, float]] = []
    named: list[tuple[str, dict[str, float]]] = []
    for i, path in enumerate(paths, 1):
        row = _measure(path)
        if row is None:
            continue
        rows.append(row)
        named.append((path.name, row))
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}…", file=sys.stderr)

    if not rows:
        print("nothing measurable", file=sys.stderr)
        return 1

    _table(args.name, rows)

    from services.lesion_gate import _MAX_EDGE_WIDTH, _MIN_CONTRAST_Z, _MIN_ON_SKIN

    print("\nagainst the shipped thresholds:")
    for column, threshold, refuse_when_below in (
        ("on_skin", _MIN_ON_SKIN, True),
        ("edge_width", _MAX_EDGE_WIDTH, False),
        ("z", _MIN_CONTRAST_Z, True),
    ):
        values = np.array([r[column] for r in rows])
        hit = values < threshold if refuse_when_below else values > threshold
        side = "below" if refuse_when_below else "above"
        print(
            f"  {column:<11s} threshold {threshold:6.2f} — "
            f"{hit.sum():4d}/{len(values)} ({100.0 * hit.mean():5.1f}%) {side} it"
        )
    print(
        "\nFor a must-PASS set every percentage above should be near zero.\n"
        "For a must-REFUSE set they are the share this gate would actually stop."
    )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", *_COLUMNS])
            for filename, row in named:
                writer.writerow([filename, *(f"{row[c]:.4f}" for c in _COLUMNS)])
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
