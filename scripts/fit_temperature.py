#!/usr/bin/env python3
"""Fit temperature scaling scalar T on validation logits (no retrain).

NOTE: Temperature scaling is NOT applied at inference — the deployed model
outputs softmax probabilities and the decision threshold in thresholds.json
was tuned on uncalibrated probs, so T has no effect on decisions.  This
script is retained for reference / future use if a logit-output model is
adopted.  Do not run it as part of the normal training pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "models" / "temperature.json"


LOGITS = ROOT / "models" / "val_logits.npz"


def main() -> None:
    try:
        import numpy as np
        from scipy.optimize import minimize_scalar
    except ImportError as e:
        print("Need numpy and scipy:", e)
        sys.exit(1)

    if not LOGITS.is_file():
        print("Missing", LOGITS, "- run the training notebook Cell 10b first to save val logits.")
        OUT.write_text(json.dumps({"T": 1.0, "fit_on": None}, indent=2), encoding="utf-8")
        return

    data = np.load(LOGITS)
    logits = data["logits"].astype(np.float64)   # log-prob surrogate from the notebook
    labels = data["labels"].astype(int)

    def nll(T: float) -> float:
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12)))

    nll_before = nll(1.0)
    res = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    t_opt = float(res.x)
    nll_after = nll(t_opt)
    OUT.write_text(
        json.dumps(
            {"T": t_opt, "fit_on": "val", "val_nll_before": nll_before, "val_nll_after": nll_after},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}  T={t_opt:.3f}  NLL {nll_before:.4f} -> {nll_after:.4f}")


if __name__ == "__main__":
    main()
