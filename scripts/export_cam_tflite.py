"""Export a two-output (last-conv feature map, softmax) TFLite for Eigen-CAM.

Same weights as production — NO retraining, and the validated classifier file
(``models/skin_classifier.tflite``) is untouched. This writes a separate
explanation-only artifact the kiosk uses for the attention overlay
(``frontend/services/eigencam.py``): the slim runtime cannot backprop, so
gradient-free Eigen-CAM over this export is the only on-device option.

Run on the training PC (needs TensorFlow):
    venv/bin/python scripts/export_cam_tflite.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "skin_classifier_full.keras"
DST = ROOT / "models" / "skin_classifier_cam.tflite"


def main() -> None:
    import tensorflow as tf

    if not SRC.is_file():
        raise SystemExit(f"missing {SRC}")
    model = tf.keras.models.load_model(str(SRC), compile=False)

    feat = None
    for layer in model.layers:
        out = getattr(layer, "output", None)
        if out is not None and len(out.shape) == 4:
            feat = out  # keep the LAST 4D output (EfficientNetB0 top activation, 7x7x1280)
    if feat is None:
        raise SystemExit("no 4D feature layer found — unexpected architecture")

    dual = tf.keras.Model(inputs=model.inputs, outputs=[feat, model.outputs[0]])
    conv = tf.lite.TFLiteConverter.from_keras_model(dual)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range: small file, float I/O
    DST.write_bytes(conv.convert())
    print(f"wrote {DST} ({DST.stat().st_size / 1e6:.1f} MB)")

    # Smoke test: shapes + latency with the same runtime family the kiosk uses.
    it = tf.lite.Interpreter(model_path=str(DST), num_threads=4)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    x = np.random.default_rng(0).uniform(0, 255, size=inp["shape"]).astype(np.float32)
    it.set_tensor(inp["index"], x)
    t0 = time.perf_counter()
    it.invoke()
    ms = (time.perf_counter() - t0) * 1000
    for od in it.get_output_details():
        print(f"output {od['name']}: shape={list(it.get_tensor(od['index']).shape)} dtype={od['dtype'].__name__}")
    print(f"single forward pass: {ms:.0f} ms (PC; expect a few x slower on Pi 4)")


if __name__ == "__main__":
    main()
