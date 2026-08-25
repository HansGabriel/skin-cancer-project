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
    "The bright areas show which parts of the photo the scanner paid most attention to. "
    "It is not an outline of the spot, and it is not a diagnosis."
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


#: Signature output holding the conv feature map (see scripts/export_cam_tflite.py).
FEATURES_OUTPUT = "features"


def _run_for_features(interpreter: Any, inp: dict, tensor: Any) -> Any:
    """Invoke the CAM model and return its conv feature map, or ``None``.

    Prefers the named signature output so the tensor is fetched by name rather
    than by output ordering, which the converter does not guarantee. Falls back
    to scanning for a 4D output so models exported before the export script
    named its outputs keep working.
    """
    try:
        runner = interpreter.get_signature_runner()
        outputs = runner(**{next(iter(runner.get_input_details())): tensor})
        feat = outputs.get(FEATURES_OUTPUT)
        if feat is not None and getattr(feat, "ndim", 0) == 4:
            return feat[0]
        for value in outputs.values():
            if getattr(value, "ndim", 0) == 4:
                return value[0]
    except Exception:  # noqa: BLE001 — pre-signature export, fall through to indices
        pass
    interpreter.set_tensor(inp["index"], tensor)
    interpreter.invoke()
    feat = None
    for od in interpreter.get_output_details():
        t = interpreter.get_tensor(od["index"])
        if t.ndim == 4:
            feat = t[0]
    return feat


def enrich_with_eigencam(result: dict[str, Any], *, want_overlay: bool = False) -> None:
    """Run the CAM export for the attention heatmap and/or the stage-3 gate.

    Silent no-op when the artifact is missing, the runtime can't load it, or the
    Keras path already produced an overlay.

    ``want_overlay`` is False on the scan path. The heatmap it produces is only
    ever shown behind a button inside "Details for staff", but building it cost
    a full second forward pass plus a full-resolution colormap blend on **every
    scan** — around a second of Pi time for a panel almost nobody opens.

    The forward pass is still made when stage 3 of the content gate is live,
    because that gate reads the same conv features and can withdraw a result.
    Skipping the pass whenever the gate is real would quietly remove a safety
    check, so the condition is "the gate cannot run" (no ``feature_stats.json``)
    — not "the heatmap is not on screen".
    """
    if result.get("attention_overlay_jpg") is not None:
        return
    rgb = result.get("rgb")
    if rgb is None or not cam_model_path().is_file():
        return
    from services.lesion_gate import ood_report, ood_stage_available

    if not want_overlay and not ood_stage_available():
        return
    try:
        from backend.streamlit_resources import get_tflite_interpreter

        it = get_tflite_interpreter(str(cam_model_path()))
        inp = it.get_input_details()[0]
        # Canonical preprocessing (backend/preprocessing.py) — same contract as production.
        tensor = to_input_tensor(rgb, inp)
        feat = _run_for_features(it, inp, tensor)
        if feat is None:
            return
        if want_overlay:
            result["attention_overlay_jpg"] = overlay_jpg(rgb, eigencam_map(feat))
            result["attention_note"] = EIGENCAM_DISCLAIMER
        # Stage 3 of the content gate rides along here rather than running in
        # services.lesion_gate: the conv features it needs are this model's
        # output, which has just been computed. Doing it in the gate would mean
        # a second full inference. ``None`` when models/feature_stats.json is absent.
        distance, threshold = ood_report(feat.mean(axis=(0, 1)))
        # Recorded whether or not the stage fires: these two numbers are the
        # only evidence available for whether the threshold suits this camera.
        result["ood_distance"] = distance
        result["ood_threshold"] = threshold
        result["out_of_distribution"] = (
            None if (distance is None or threshold is None) else distance > threshold
        )
    except Exception:  # noqa: BLE001 — explanation is optional; never break a scan
        return
