"""Run the kiosk's classifier over one split of training/splits.csv.

Shared by calibrate.py (validation split) and eval.py (test split), so both read the
same file the same way and score images through dermascan.classifier - the exact
code, model file and 4-view TTA the kiosk uses.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dermascan.classifier import Classifier  # noqa: E402

SPLITS_CSV = ROOT / "training" / "splits.csv"


def read_split(csv_path: Path, split: str) -> list[dict]:
    """Rows of one split. A CSV without a `split` column is taken whole."""
    if not Path(csv_path).is_file():
        sys.exit(f"{csv_path} does not exist yet. Run the training notebook first (Cell 5 writes it).")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "split" in rows[0]:
        rows = [r for r in rows if r["split"] == split]
    if not rows:
        sys.exit(f"No '{split}' rows in {csv_path}.")
    return rows


def score_rows(
    rows: list[dict], clf: Classifier, images_dir: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(true label index per image, raw TTA-averaged probabilities per image).

    Stops if any image cannot be read: a metric over a silently shorter list is wrong.
    """
    lookup = {p.stem: p for p in Path(images_dir).rglob("*.jpg")} if images_dir else {}
    truth, probs, missing = [], [], []
    for k, r in enumerate(rows, 1):
        path = lookup.get(r["image_id"]) if lookup else Path(r["path"])
        bgr = cv2.imread(str(path)) if path else None
        if bgr is None:
            missing.append(r["image_id"])
            continue
        p = clf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        truth.append(int(r["label_idx"]))
        probs.append(p.raw_probs)
        if k % 200 == 0:
            print(f"  {k}/{len(rows)}", file=sys.stderr)
    if missing:
        sys.exit(f"{len(missing)} images could not be read (first: {missing[:3]}). Use --images-dir.")
    return np.asarray(truth), np.asarray(probs, dtype=np.float64)
