#!/usr/bin/env python3
"""Re-export int8 TFLite from existing Keras (no retrain).

Full-integer quantization REQUIRES a representative dataset — without it the
converter silently falls back to dynamic-range quant (which is why the previous
"int8" file was the same ~5MB as the float model). Mirrors training notebook
Cell 13: ~200 preprocessed float32 [0,255] images sampled from the test split.

Usage:
    ./venv/bin/python scripts/export_tflite_int8.py
    # then check parity before adopting:
    ./venv/bin/python scripts/eval_threshold.py --model models/skin_classifier_int8.tflite
Accept only if screening sensitivity is within 1pt of the float model, then
benchmark on the Pi and switch via SKIN_MODEL=skin_classifier_int8.tflite.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERAS = ROOT / "models" / "skin_classifier_full.keras"
TEST_CSV = ROOT / "models" / "test_split.csv"
OUT = ROOT / "models" / "skin_classifier_int8.tflite"

IMAGE_SIZE = 224
N_CALIB = 200


def main() -> None:
    if not KERAS.is_file():
        print("Missing", KERAS)
        sys.exit(1)
    if not TEST_CSV.is_file():
        print("Missing", TEST_CSV, "- representative dataset requires it")
        sys.exit(1)

    import cv2
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    df = pd.read_csv(TEST_CSV)
    paths = [p for p in df["path"].tolist() if Path(p).is_file()]
    if len(paths) < 50:
        print(f"Only {len(paths)} readable calibration images — need >=50. Check test_split.csv paths.")
        sys.exit(1)
    # Deterministic spread across the split (all classes represented).
    step = max(1, len(paths) // N_CALIB)
    calib_paths = paths[::step][:N_CALIB]
    print(f"Calibrating activations on {len(calib_paths)} images")

    def representative_dataset():
        for p in calib_paths:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            # Training contract: float32 [0,255], EfficientNet normalizes internally.
            yield [np.expand_dims(img.astype(np.float32), axis=0)]

    model = tf.keras.models.load_model(str(KERAS), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    OUT.write_bytes(tflite_model)
    print("Wrote", OUT, len(tflite_model), "bytes")
    float_path = ROOT / "models" / "skin_classifier.tflite"
    if float_path.is_file():
        ratio = len(tflite_model) / float_path.stat().st_size
        print(f"Size vs float model: {ratio:.2f}x "
              f"({'true int8' if ratio < 0.6 else 'WARNING: looks like dynamic-range fallback'})")


if __name__ == "__main__":
    main()
