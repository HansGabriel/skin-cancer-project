"""Gradient-free attention overlay (Eigen-CAM) for the kiosk.

The slim TFLite runtime cannot backprop, so Grad-CAM is impossible on the Pi.
Eigen-CAM (Muhammad & Yeasin 2020) needs only ONE forward pass: it projects the
last conv feature map onto its first principal component — class-agnostic, no
gradients. It runs against ``models/skin_classifier_cam.tflite``, a two-output
re-export of the production weights (see ``scripts/export_cam_tflite.py`` — no
retraining). When that artifact is absent this module is a silent no-op and the
UI simply shows no heatmap panel, same as before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.preprocessing import to_input_tensor

# Same wording as the Keras Grad-CAM path — the UI shows one disclaimer either way.
EIGENCAM_DISCLAIMER = (
    "Heatmap highlights regions the model weighted most; it is not lesion ground truth or a diagnosis."
)


def cam_model_path() -> Path:
    p = os.environ.get("SKIN_CAM_MODEL_PATH")
    if p:
        return Path(p)
    return Path(__file__).resolve().parent.parent.parent / "models" / "skin_classifier_cam.tflite"


def eigencam_map(activations: np.ndarray) -> np.ndarray:
    """(H, W, C) feature map → (H, W) attention in [0, 1].

    First principal component of the (positions × channels) activation matrix,
    sign-corrected so the component points at the dominant activation mass,
    rectified and min-max normalized.
    """
    h, w, c = activations.shape
    a = activations.reshape(h * w, c).astype(np.float64)
    a -= a.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(a, full_matrices=False)
    proj = a @ vt[0]
    if abs(float(proj.min())) > abs(float(proj.max())):
        proj = -proj
    proj = np.maximum(proj, 0.0)
    rng = float(proj.max() - proj.min())
    if rng < 1e-12:
        return np.zeros((h, w), dtype=np.float32)
    return ((proj - proj.min()) / rng).reshape(h, w).astype(np.float32)


def overlay_jpg(rgb: np.ndarray, cam01: np.ndarray) -> bytes:
    """JET-colormap 50/50 blend at the capture's resolution (mirrors the Grad-CAM look)."""
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam01, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    blend = np.clip(0.5 * rgb.astype(np.float32) + 0.5 * heat_rgb.astype(np.float32), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(blend, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def enrich_with_eigencam(result: dict[str, Any]) -> None:
    """Fill ``gradcam_overlay_jpg`` (the slot the Keras path uses) from the CAM export.

    Silent no-op when the artifact is missing, the runtime can't load it, or the
    Keras path already produced an overlay.
    """
    if result.get("gradcam_overlay_jpg") is not None:
        return
    rgb = result.get("rgb")
    if rgb is None or not cam_model_path().is_file():
        return
    try:
        from backend.streamlit_resources import get_tflite_interpreter

        it = get_tflite_interpreter(str(cam_model_path()))
        inp = it.get_input_details()[0]
        # Canonical preprocessing (backend/preprocessing.py) — same contract as production.
        it.set_tensor(inp["index"], to_input_tensor(rgb, inp))
        it.invoke()
        feat = None
        for od in it.get_output_details():
            t = it.get_tensor(od["index"])
            if t.ndim == 4:
                feat = t[0]
        if feat is None:
            return
        result["gradcam_overlay_jpg"] = overlay_jpg(rgb, eigencam_map(feat))
        result["gradcam_disclaimer"] = EIGENCAM_DISCLAIMER
    except Exception:  # noqa: BLE001 — explanation is optional; never break a scan
        return
