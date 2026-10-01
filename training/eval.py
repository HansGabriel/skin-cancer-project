"""Score the deployed model on the held-out test split, exactly as the kiosk runs it.

    python training/eval.py                      # training/splits.csv, rows where split == test
    python training/eval.py --csv training/test_split_image_level.csv   # the OLD leaky split
    python training/eval.py --images-dir /path/to/ham10000               # re-root the image paths

Uses dermascan.classifier.Classifier - the same 4-view TTA, the same threshold,
the same temperature - so the numbers describe the device, not a re-analysis.
Prints cancer sensitivity / specificity at the deployed threshold, the 3x3
confusion matrix, and the argmax view for comparison. Writes docs/METRICS.md's
table to stdout so it can be pasted in.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dermascan.classifier import FLAGGED_LABELS, Classifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "training" / "splits.csv"))
    ap.add_argument("--images-dir", default=None, help="folder tree holding the ISIC_*.jpg files, if the CSV paths are stale")
    ap.add_argument("--no-tta", action="store_true", help="single view instead of the kiosk's four")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    if rows and "split" in rows[0]:
        rows = [r for r in rows if r["split"] == "test"]
    if not rows:
        print("no test rows in", args.csv)
        return 1

    lookup: dict[str, Path] = {}
    if args.images_dir:
        lookup = {p.stem: p for p in Path(args.images_dir).rglob("*.jpg")}

    clf = Classifier(use_tta=not args.no_tta)
    labels = clf.labels
    n = len(labels)
    cm_thr = np.zeros((n, n), int)
    cm_arg = np.zeros((n, n), int)
    missing = []
    for k, r in enumerate(rows, 1):
        path = lookup.get(r["image_id"]) if lookup else Path(r["path"])
        bgr = cv2.imread(str(path)) if path else None
        if bgr is None:
            missing.append(r["image_id"])
            continue
        p = clf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        truth = int(r["label_idx"])
        cm_thr[truth, labels.index(p.label)] += 1
        cm_arg[truth, int(np.argmax(p.raw_probs))] += 1
        if k % 100 == 0:
            print(f"  {k}/{len(rows)}", file=sys.stderr)
    if missing:
        print(f"{len(missing)} images could not be read (first: {missing[:3]}); metrics would be wrong, stopping.")
        return 1

    flagged = [labels.index(l) for l in FLAGGED_LABELS if l in labels]
    benign = [i for i in range(n) if i not in flagged]

    def screen(cm):
        cancer_rows = cm[flagged].sum(axis=0)
        benign_rows = cm[benign].sum(axis=0)
        sens = cancer_rows[flagged].sum() / max(1, cancer_rows.sum())
        spec = benign_rows[benign].sum() / max(1, benign_rows.sum())
        return sens, spec

    def show(cm, title):
        print(f"\n{title}  (n={cm.sum()}, accuracy {np.trace(cm) / cm.sum():.3f})")
        print(" " * 15 + "".join(f"{l[:12]:>14}" for l in labels))
        for i, l in enumerate(labels):
            print(f"{l:>15}" + "".join(f"{v:>14}" for v in cm[i]))
        s, sp = screen(cm)
        print(f"cancer sensitivity {s:.3f}   benign specificity {sp:.3f}")
        for i, l in enumerate(labels):
            print(f"  {l} recall {cm[i, i] / max(1, cm[i].sum()):.3f}")

    show(cm_thr, f"At the deployed threshold {clf.thresholds.get('screen_cancer_threshold')} with {'1 view' if args.no_tta else '4-view TTA'}")
    show(cm_arg, "Argmax (no screening threshold), for comparison")
    s, sp = screen(cm_thr)
    print("\nFor docs/METRICS.md:")
    print(f"| Cancer sensitivity | {s:.3f} |\n| Benign specificity | {sp:.3f} |\n| Accuracy (3-class, at threshold) | {np.trace(cm_thr) / cm_thr.sum():.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
