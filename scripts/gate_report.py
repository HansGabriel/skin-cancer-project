"""Run a folder of photos through the photo gate and print what it measured.

    python scripts/gate_report.py FOLDER [FOLDER ...]
    python scripts/gate_report.py tests/fixtures/gate/refuse --expect refuse
    python scripts/gate_report.py captures/ --full --csv /tmp/gate.csv

One row per photo: the file, the refusal code (or "pass"), then every number the
gate decided on. "-" means the gate stopped before measuring it; --full runs every
measurement regardless, which is what you want when setting a limit from real
photos (the limits in dermascan/config.py marked GUESSED are waiting for this).
--expect lists the photos that disagree and exits 1, so it doubles as a check.

Works on a Pi's SAVE_CAPTURES_DIR folder. The classifier is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dermascan  # noqa: E402,F401  (sets OpenCV's decode limit before cv2 loads)
from dermascan import gate, scan  # noqa: E402

COLUMNS = (
    "skin_fraction", "focus", "brightness", "spot_fraction", "spot_solidity", "spot_contrast",
    "spot_offset", "spot_at_edge", "other_marks", "spot_dominance", "ring_skin", "edge_width",
)
SHORT = ("skin", "focus", "bright", "spot", "solid", "contr", "offset", "edge?", "others", "domin", "ring", "edge_w")
EXTENSIONS = (".jpg", ".jpeg", ".png")


def photos_in(folders: list[Path]) -> list[Path]:
    out: list[Path] = []
    for folder in folders:
        if folder.is_file():
            out.append(folder)
        else:
            out += sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    return out


def measure(path: Path, full: bool) -> tuple[str, dict[str, float]]:
    checked = scan.check_photo(path.read_bytes())
    if checked.report is None:
        return "not_a_picture", {}
    code = checked.report.refusal.code if checked.report.refusal else "pass"
    measured = gate.measure_everything(checked.rgb) if full else checked.report.measured
    return code, measured


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--expect", choices=("pass", "refuse"), help="exit 1 if any photo disagrees")
    ap.add_argument("--full", action="store_true", help="measure everything, even after a refusal")
    ap.add_argument("--csv", type=Path, help="also write the rows here")
    args = ap.parse_args()

    paths = photos_in(args.folders)
    if not paths:
        raise SystemExit("no .jpg/.jpeg/.png files found")

    rows: list[tuple[str, str, dict[str, float]]] = []
    print(f"{'file':<32} {'result':<14} " + " ".join(f"{s:>7}" for s in SHORT))
    for path in paths:
        code, measured = measure(path, args.full)
        rows.append((path.name, code, measured))
        cells = " ".join(f"{measured[c]:>7.3f}" if c in measured else f"{'-':>7}" for c in COLUMNS)
        print(f"{path.name[:32]:<32} {code:<14} {cells}")

    counts = Counter(code for _, code, _ in rows)
    print("\n" + "  ".join(f"{code}: {n}" for code, n in sorted(counts.items())) + f"  (of {len(rows)})")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("file", "result", *COLUMNS))
            for name, code, measured in rows:
                w.writerow((name, code, *(measured.get(c, "") for c in COLUMNS)))
        print(f"wrote {args.csv}")

    if args.expect:
        wrong = [
            name for name, code, _ in rows
            if code == "not_a_picture" or (code == "pass") != (args.expect == "pass")
        ]
        if wrong:
            print(f"\n{len(wrong)} photo(s) did not {args.expect}: " + ", ".join(wrong))
            return 1
        print(f"\nall {len(rows)} photos {args.expect} as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
