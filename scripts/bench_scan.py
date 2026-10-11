"""Time the scan path the way the kiosk runs it, and print one line per stage.

    python scripts/bench_scan.py                      # synthetic lesion photos (no dataset needed)
    python scripts/bench_scan.py --images captures/   # real captures, if SAVE_CAPTURES_DIR ever kept some
    python scripts/bench_scan.py --rounds 20

Each round does what one visitor does: run_check (decode + gate) and then run_scan
on the same bytes, first with the check reused as the kiosk does, then from scratch
as the web demo does. Medians in milliseconds, so one slow round does not skew it.
Run it on the Pi from the project venv; paste the output into docs/METRICS.md.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))  # the synthetic photos the tests use

from dermascan import scan  # noqa: E402
from dermascan.classifier import get_classifier  # noqa: E402


def synthetic_photos() -> list[bytes]:
    from conftest import SKIN_TONES, jpeg, textured, with_lesion

    return [jpeg(with_lesion(textured(tone, size=1024))) for tone in SKIN_TONES.values()]


def load_photos(folder: Path | None) -> list[bytes]:
    if folder is None:
        return synthetic_photos()
    photos = [p.read_bytes() for p in sorted(folder.glob("*.jpg"))]
    if not photos:
        raise SystemExit(f"no .jpg files in {folder}")
    return photos


def median_of(rows: list[dict[str, int]], key: str) -> int:
    return int(statistics.median(r.get(key, 0) for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=None, help="folder of .jpg photos (default: synthetic)")
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()

    photos = load_photos(args.images)
    get_classifier()  # load the model once, outside the timing, as kiosk/server.py does
    for p in photos:  # warm up: first calls pay for lazy imports and cache fills
        scan.run_scan(p)

    reused: list[dict[str, int]] = []
    cold: list[dict[str, int]] = []
    checks: list[dict[str, int]] = []
    for _ in range(args.rounds):
        for p in photos:
            checked = scan.check_photo(p)
            checks.append(checked.stage_ms)
            reused.append(scan.run_scan(p, checked=checked).stage_ms)
            cold.append(scan.run_scan(p).stage_ms)

    def line(name: str, rows: list[dict[str, int]], keys: tuple[str, ...]) -> None:
        parts = " ".join(f"{k}={median_of(rows, k)}" for k in keys)
        total = int(statistics.median(sum(r.values()) for r in rows))
        print(f"{name:<22} ms {parts} total={total}")

    print(f"{len(photos)} photos x {args.rounds} rounds, 4-view TTA, medians:")
    line("check (decode+gate)", checks, ("decode", "gate"))
    line("scan after check", reused, ("decode", "gate", "model", "signs"))
    line("scan from scratch", cold, ("decode", "gate", "model", "signs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
