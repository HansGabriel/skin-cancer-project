"""Fit the screening threshold and the temperature on the model the kiosk will run.

    python training/calibrate.py                 # validation rows of training/splits.csv
    python training/calibrate.py --target 0.90   # cancer sensitivity to guarantee on validation

Scores every validation image through dermascan.classifier (the exported .tflite,
4-view TTA), then writes:

    models/thresholds.json   the highest threshold on p(pre_cancerous)+p(malignant) that
                             still flags at least --target of validation cancers
    models/temperature.json  T minimising validation log-loss of softmax(log(p) / T)

Fitting on the device path matters: the quantised TFLite model with TTA does not give
the Keras model's probabilities, and the threshold is small enough (~0.1) to move.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from split_scores import ROOT, SPLITS_CSV, read_split, score_rows

sys.path.insert(0, str(ROOT))
from dermascan import config  # noqa: E402
from dermascan.classifier import Classifier, load_labels  # noqa: E402


def fit_threshold(truth: np.ndarray, probs: np.ndarray, flagged_idx: list[int], target: float) -> dict:
    """Highest threshold whose validation cancer sensitivity is still >= target."""
    is_cancer = np.isin(truth, flagged_idx)
    p_cancer = probs[:, flagged_idx].sum(axis=1)
    best = None
    for thr in np.round(np.arange(0.005, 0.951, 0.005), 3):
        flagged = p_cancer >= thr
        sens = float((flagged & is_cancer).sum() / max(1, is_cancer.sum()))
        spec = float((~flagged & ~is_cancer).sum() / max(1, (~is_cancer).sum()))
        if sens >= target:
            best = (float(thr), sens, spec)
    if best is None:
        sys.exit(f"No threshold reaches {target:.0%} sensitivity on validation; the model needs work.")
    return {"screen_cancer_threshold": best[0], "val_sensitivity": best[1], "val_specificity": best[2]}


def fit_temperature(truth: np.ndarray, probs: np.ndarray) -> dict:
    """T in [0.5, 5] minimising negative log-likelihood, by a fine grid (no scipy needed)."""
    logits = np.log(np.clip(probs, 1e-7, 1.0))

    def nll(t: float) -> float:
        z = logits / t
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(p[np.arange(len(truth)), truth] + 1e-12)))

    grid = np.arange(0.5, 5.0001, 0.005)
    t = float(grid[int(np.argmin([nll(g) for g in grid]))])
    return {"T": round(t, 4), "val_nll_before": nll(1.0), "val_nll_after": nll(t)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(SPLITS_CSV))
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--target", type=float, default=0.90, help="cancer sensitivity to guarantee on validation")
    args = ap.parse_args()

    labels = load_labels(config.LABELS_PATH)
    flagged_idx = [labels.index("pre_cancerous"), labels.index("malignant")]
    rows = read_split(args.csv, "val")
    clf = Classifier(use_tta=True)
    print(f"Scoring {len(rows)} validation images with {clf.model_path.name}, 4-view TTA...")
    truth, probs = score_rows(rows, clf, args.images_dir)

    thr = fit_threshold(truth, probs, flagged_idx, args.target)
    temp = fit_temperature(truth, probs)

    thresholds = {
        **thr,
        "target_sensitivity": args.target,
        "fit_on": "val, exported tflite, 4-view TTA (training/calibrate.py)",
        "labels": labels,
        "precancer_idx": flagged_idx[0],
        "malignant_idx": flagged_idx[1],
        "decision": "flag cancerous if p(pre_cancerous)+p(malignant) >= screen_cancer_threshold",
    }
    config.THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2) + "\n")
    config.TEMPERATURE_PATH.write_text(json.dumps({**temp, "fit_on": thresholds["fit_on"]}, indent=2) + "\n")
    print(
        f"threshold {thr['screen_cancer_threshold']:.3f}: validation sensitivity {thr['val_sensitivity']:.3f}, "
        f"specificity {thr['val_specificity']:.3f}\n"
        f"temperature T={temp['T']:.3f}: log-loss {temp['val_nll_before']:.4f} -> {temp['val_nll_after']:.4f}\n"
        f"wrote {config.THRESHOLDS_PATH.name} and {config.TEMPERATURE_PATH.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
