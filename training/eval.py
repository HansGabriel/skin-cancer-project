"""Score the deployed model on the held-out test split, exactly as the kiosk runs it.

    python training/eval.py                      # test rows of training/splits.csv
    python training/eval.py --csv training/test_split_image_level.csv   # the OLD leaky split
    python training/eval.py --images-dir /path/to/ham10000               # re-root stale image paths
    python training/eval.py --no-tta             # single view instead of the kiosk's four

Uses dermascan.classifier - the same model file, 4-view TTA, threshold and
temperature - so the numbers describe the device, not a re-analysis. Prints the
3x3 confusion matrix at the deployed threshold and at argmax, and the rows to
paste into docs/METRICS.md. Run training/calibrate.py first after a retrain.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from split_scores import SPLITS_CSV, read_split, score_rows

from dermascan.classifier import FLAGGED_LABELS, Classifier, decide_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(SPLITS_CSV))
    ap.add_argument("--images-dir", default=None, help="folder tree holding the ISIC_*.jpg files")
    ap.add_argument("--no-tta", action="store_true", help="single view instead of the kiosk's four")
    args = ap.parse_args()

    rows = read_split(args.csv, "test")
    clf = Classifier(use_tta=not args.no_tta)
    labels, n = clf.labels, len(clf.labels)
    print(f"Scoring {len(rows)} test images with {clf.model_path.name}, {'1 view' if args.no_tta else '4-view TTA'}...")
    truth, probs = score_rows(rows, clf, args.images_dir)

    cm_thr = np.zeros((n, n), int)
    cm_arg = np.zeros((n, n), int)
    for t, p in zip(truth, probs):
        cm_thr[t, decide_index(p, clf.thresholds)] += 1
        cm_arg[t, int(np.argmax(p))] += 1

    flagged = [labels.index(l) for l in FLAGGED_LABELS]
    benign = [i for i in range(n) if i not in flagged]

    def screen(cm: np.ndarray) -> tuple[float, float]:
        cancer, clean = cm[flagged].sum(axis=0), cm[benign].sum(axis=0)
        return cancer[flagged].sum() / max(1, cancer.sum()), clean[benign].sum() / max(1, clean.sum())

    def show(cm: np.ndarray, title: str) -> None:
        print(f"\n{title}  (n={cm.sum()}, accuracy {np.trace(cm) / cm.sum():.3f})")
        print(" " * 15 + "".join(f"{l[:12]:>14}" for l in labels) + "   <- predicted")
        for i, l in enumerate(labels):
            print(f"{l:>15}" + "".join(f"{v:>14}" for v in cm[i]))
        sens, spec = screen(cm)
        print(f"cancer sensitivity {sens:.3f}   benign specificity {spec:.3f}")
        print("   " + "   ".join(f"{l} recall {cm[i, i] / max(1, cm[i].sum()):.3f}" for i, l in enumerate(labels)))

    show(cm_thr, f"At the deployed threshold {clf.thresholds.get('screen_cancer_threshold')}")
    show(cm_arg, "Argmax (no screening threshold), for comparison")
    sens, spec = screen(cm_thr)
    sens_a, spec_a = screen(cm_arg)
    print("\nFor docs/METRICS.md:")
    print(f"| Cancer sensitivity (at threshold) | {sens:.3f} |")
    print(f"| Benign specificity (at threshold) | {spec:.3f} |")
    print(f"| 3-class accuracy (at threshold) | {np.trace(cm_thr) / cm_thr.sum():.3f} |")
    print(f"| Argmax accuracy | {np.trace(cm_arg) / cm_arg.sum():.3f} |")
    for i, l in enumerate(labels):
        print(f"| {l} recall (argmax) | {cm_arg[i, i] / max(1, cm_arg[i].sum()):.3f} |")
    print(f"| Cancer sensitivity / benign specificity (argmax) | {sens_a:.3f} / {spec_a:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
