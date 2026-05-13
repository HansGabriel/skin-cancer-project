#!/usr/bin/env python3
"""Verify a Keras model for PC Grad-CAM / optional 7-class differential.

The training notebook saves ``models/skin_classifier_full.keras`` (3-class softmax today).
For 7-class bars in Streamlit, retrain with seven HAM10000 dx outputs in this order::

    akiec, bcc, bkl, df, mel, nv, vasc

Then set ``SKIN_KERAS_PATH`` to that file (see Settings in the app).

Usage::

    python scripts/export_keras_for_vis.py --verify path/to/model.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Verify Keras model for DermaScan v2 vis.")
    p.add_argument(
        "--verify",
        type=Path,
        metavar="MODEL.keras",
        help="Load model and print output shape (requires TensorFlow).",
    )
    args = p.parse_args()
    if not args.verify:
        print(__doc__)
        return 0
    path = args.verify
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1
    import tensorflow as tf

    m = tf.keras.models.load_model(str(path), compile=False)
    out = m.output_shape
    print("Loaded:", path.resolve())
    print("Output shape:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
