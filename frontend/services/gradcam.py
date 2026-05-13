"""Grad-CAM++ and optional 7-class softmax from a full Keras model (PC only; lazy TF import)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Order must match the 7-class training head if you export a 7-way model.
HAM7_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

GRAD_CAM_DISCLAIMER = (
    "Heatmap highlights regions the model weighted most; it is not lesion ground truth or a diagnosis."
)


def load_keras_model(path: str | Path) -> Any:
    import tensorflow as tf

    return tf.keras.models.load_model(str(path), compile=False)


def _softmax_probs(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64) - np.max(logits)
    e = np.exp(x)
    return (e / np.sum(e)).astype(np.float32)


def predict_probs(model: Any, rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (prob_vector, argmax_index) for model's softmax output."""
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    x224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    xb = np.expand_dims(preprocess_input(x224.astype(np.float32)), axis=0)
    raw = model.predict(xb, verbose=0)[0].astype(np.float64)
    s = float(np.sum(raw))
    if s <= 0 or np.any(raw < 0) or not (0.95 <= s <= 1.05):
        probs = _softmax_probs(raw)
    else:
        probs = raw.astype(np.float32)
    return probs, int(np.argmax(probs))


def gradcam_overlay_jpg(model: Any, rgb: np.ndarray, class_idx: int) -> tuple[np.ndarray, bytes]:
    """Return (heatmap_0_1 at original resolution, jpeg bytes RGB overlay)."""
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.utils.scores import CategoricalScore

    h, w = rgb.shape[:2]
    x224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    xb = np.expand_dims(preprocess_input(x224.astype(np.float32)), axis=0)

    gradcam = GradcamPlusPlus(model, clone=True)
    cam = gradcam(CategoricalScore(class_idx), xb, penultimate_layer=None)
    if isinstance(cam, list):
        cam = cam[0]
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim > 2:
        cam = cam.squeeze()
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cmin, cmax = float(cam.min()), float(cam.max())
    if cmax - cmin < 1e-8:
        norm = np.zeros_like(cam)
    else:
        norm = (cam - cmin) / (cmax - cmin)
    heat_u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    blend = np.clip(0.5 * rgb.astype(np.float32) + 0.5 * heat_rgb.astype(np.float32), 0, 255).astype(
        np.uint8
    )
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(blend, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return norm, enc.tobytes()


def enrich_result_with_vis(result: dict[str, Any], rgb: np.ndarray, model: Any) -> None:
    """Populate ``seven_class_probs``, ``gradcam_overlay_jpg`` using an already-loaded ``model``."""
    try:
        probs, idx = predict_probs(model, rgb)
    except Exception:  # noqa: BLE001
        return
    n = int(probs.shape[0])
    if n == len(HAM7_LABELS):
        result["seven_class_probs"] = {HAM7_LABELS[i]: float(probs[i]) * 100.0 for i in range(n)}
    else:
        result["seven_class_probs"] = None
    try:
        _, jpg = gradcam_overlay_jpg(model, rgb, idx)
        result["gradcam_overlay_jpg"] = jpg
        result["gradcam_disclaimer"] = GRAD_CAM_DISCLAIMER
    except Exception:  # noqa: BLE001
        result["gradcam_overlay_jpg"] = None
        result["gradcam_disclaimer"] = GRAD_CAM_DISCLAIMER


def attach_vis_extras(
    result: dict[str, Any],
    rgb: np.ndarray,
    keras_path: str,
) -> None:
    """Load model from disk (no caching — prefer ``enrich_result_with_vis`` + ``st.cache_resource``)."""
    p = Path(keras_path)
    if not p.is_file():
        return
    try:
        model = load_keras_model(p)
    except Exception:  # noqa: BLE001
        return
    enrich_result_with_vis(result, rgb, model)
