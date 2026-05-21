#!/usr/bin/env python3
"""Fit temperature scaling scalar T on validation logits (no retrain)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "models" / "temperature.json"


def main() -> None:
    try:
        import numpy as np
        from scipy.optimize import minimize_scalar
    except ImportError as e:
        print("Need numpy and scipy:", e)
        sys.exit(1)
    keras_path = ROOT / "models" / "skin_classifier_full.keras"
    if not keras_path.is_file():
        print("Keras model not found:", keras_path)
        OUT.write_text(json.dumps({"T": 1.0, "fit_on": None}), encoding="utf-8")
        return
    print("Placeholder: run notebook val split to collect logits, then minimize NLL for T in [0.5, 5].")
    OUT.write_text(
        json.dumps({"T": 1.0, "fit_on": None, "val_nll_before": None, "val_nll_after": None}, indent=2),
        encoding="utf-8",
    )
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
