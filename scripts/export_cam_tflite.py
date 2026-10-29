"""Export a separate Eigen-CAM feature model next to a freshly trained classifier.

NOT needed since 2026-10-29: dermascan/cam.py reads the same feature map out of
models/skin_classifier.tflite itself. Keep this for the day a quantised export
fuses that layer away; cam.py prefers this file when it is present and its .json
names the deployed classifier's sha256.

Run on the training PC, in the SAME run that wrote models/skin_classifier.tflite
(by hand, right after the notebook's export cell - the notebook does not call it):

    venv/bin/python scripts/export_cam_tflite.py
    venv/bin/python scripts/export_cam_tflite.py --keras training/skin_classifier_full.keras --out /tmp/cam

Writes two files:
    skin_classifier_cam.tflite  - one output, EfficientNetB0's last conv map (7x7x1280),
                                  dynamic-range quantised (small file, float in/out)
    skin_classifier_cam.json    - {"classifier_sha256", "layer", "keras"}; dermascan/cam.py
                                  refuses the CAM file unless this sha is the deployed
                                  classifier's, so a heatmap from other weights never ships.

No probabilities are exported: the classifier is int8 and this file is not, so
their numbers would differ. The kiosk takes probabilities from the classifier only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Most specific first: "top_activation" is EfficientNetB0's final conv activation.
CAM_LAYER_CANDIDATES = ("top_activation", "top_conv", "out_relu")


def find_feature_layer(model, candidates=CAM_LAYER_CANDIDATES):
    """The first named 4-D layer output, or raise with the names tried."""
    for name in candidates:
        try:
            out = model.get_layer(name).output
        except ValueError:
            continue
        if len(out.shape) == 4:
            return out, name
    raise SystemExit(f"none of {candidates} is a 4-D layer; check model.summary() and pass --layer")


def export(keras_path: Path, classifier_path: Path, out_dir: Path, layer: str | None = None) -> Path:
    import tensorflow as tf

    model = tf.keras.models.load_model(str(keras_path), compile=False)
    feat, name = find_feature_layer(model, (layer,) if layer else CAM_LAYER_CANDIDATES)
    cam_model = tf.keras.Model(inputs=model.inputs, outputs=feat)
    conv = tf.lite.TFLiteConverter.from_keras_model(cam_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range: weights int8, activations float
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "skin_classifier_cam.tflite"
    dst.write_bytes(conv.convert())
    meta = {
        "classifier_sha256": hashlib.sha256(classifier_path.read_bytes()).hexdigest(),
        "layer": name,
        "keras": keras_path.name,
    }
    (out_dir / "skin_classifier_cam.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB) from layer '{name}' for {classifier_path.name}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keras", type=Path, default=ROOT / "training" / "skin_classifier_full.keras")
    ap.add_argument("--classifier", type=Path, default=ROOT / "models" / "skin_classifier.tflite")
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    ap.add_argument("--layer", default=None, help="conv layer for the heatmap (default: first found of %s)" % (CAM_LAYER_CANDIDATES,))
    a = ap.parse_args()
    for p in (a.keras, a.classifier):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    export(a.keras, a.classifier, a.out, a.layer)


if __name__ == "__main__":
    main()
