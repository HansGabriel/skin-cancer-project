#!/usr/bin/env python3
"""Re-export int8 TFLite from existing Keras (no retrain)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERAS = ROOT / "models" / "skin_classifier_full.keras"
OUT = ROOT / "models" / "skin_classifier_int8.tflite"


def main() -> None:
    if not KERAS.is_file():
        print("Missing", KERAS)
        sys.exit(1)
    import tensorflow as tf

    model = tf.keras.models.load_model(str(KERAS), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    OUT.write_bytes(tflite_model)
    print("Wrote", OUT, len(tflite_model), "bytes")


if __name__ == "__main__":
    main()
