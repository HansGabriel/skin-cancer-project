"""Eigen-CAM: "where the scanner looked", for the staff drawer only.

    photo (RGB, <= 1024 px)
      -> resize to 224, same raw [0, 255] input as the classifier
      -> models/skin_classifier_cam.tflite  (one output: EfficientNetB0's last conv
                                             map, 7 x 7 x 1280 - no probabilities)
      -> eigencam_map   first principal component of that map, 7 x 7 in [0, 1]
      -> overlay_jpg    stretched over the photo, JET colours, half-and-half blend

Why Eigen-CAM and not Grad-CAM: the Pi's TFLite runtime has no gradients, so a
heatmap needs a method that works from one forward pass (Muhammad & Yeasin 2020).
It is class-agnostic: it shows what the network found salient, not "why cancer".

The CAM file is exported by scripts/export_cam_tflite.py from the SAME training
run as models/skin_classifier.tflite; skin_classifier_cam.json records that
classifier's sha256. get_cam() returns None (button hidden) when the file is
missing or the sha does not match, so a heatmap from other weights never ships.

Display only. Nothing here feeds the verdict. About a second on a Pi 4, run on
demand when staff press the button, never on the scan path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from dermascan import config
from dermascan.classifier import _interpreter_class, to_input_tensor

log = logging.getLogger("dermascan.cam")

# Shown with every heatmap. Staff-facing, but still plain words.
CAM_NOTE = "Bright areas are where the scanner looked. Not an outline of the spot, not a diagnosis."


def eigencam_map(activations: np.ndarray) -> np.ndarray:
    """(H, W, C) feature map -> (H, W) attention in [0, 1].

    First principal component of the (positions x channels) matrix, sign-corrected
    so it points at the dominant activation mass, rectified and min-max scaled.
    A flat map (nothing salient) gives all zeros.
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


def overlay_jpg(rgb: np.ndarray, cam01: np.ndarray, alpha: float = config.CAM_ALPHA) -> bytes:
    """The heatmap stretched over the photo: JET colours, `alpha` heat on (1 - alpha) photo."""
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam01, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    blend = np.clip((1 - alpha) * rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(blend, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), config.CAM_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cam_matches_classifier(meta_path: Path = config.CAM_META_PATH, model_path: Path = config.MODEL_PATH) -> bool:
    """True only when the CAM file's recorded classifier sha is the deployed classifier's."""
    try:
        recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("classifier_sha256", "")
    except (OSError, ValueError):
        return False
    return bool(recorded) and model_path.is_file() and recorded == sha256_of(model_path)


class Cam:
    """The CAM interpreter. One per process; `heatmap` is safe to call from any thread."""

    def __init__(self, path: Path = config.CAM_PATH, num_threads: int = config.NUM_THREADS) -> None:
        self.model_path = path
        self._it = _interpreter_class()(model_path=str(path), num_threads=num_threads)
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]
        outs = [o for o in self._it.get_output_details() if len(o["shape"]) == 4]
        if not outs:
            raise ValueError(f"{path.name} has no 4-D (feature map) output")
        self._out = outs[0]
        self._lock = threading.Lock()

    def attention(self, rgb: np.ndarray) -> np.ndarray:
        """(7, 7) map in [0, 1] for one photo (single view, no TTA)."""
        small = cv2.resize(rgb, (config.INPUT_SIZE, config.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        x = to_input_tensor(small, self._in)
        with self._lock:
            self._it.set_tensor(self._in["index"], x)
            self._it.invoke()
            feat = self._it.get_tensor(self._out["index"])[0].astype(np.float32)
        return eigencam_map(feat)

    def heatmap(self, rgb: np.ndarray) -> bytes:
        """The overlay JPEG for one photo."""
        return overlay_jpg(rgb, self.attention(rgb))


_cam: Cam | None = None
_cam_tried = False
_cam_lock = threading.Lock()


def get_cam() -> Cam | None:
    """The shared Cam, or None when there is no CAM file for THIS classifier.

    Tried once per process: a missing file is normal (the button stays hidden),
    a mismatched sha gets one warning in the log.
    """
    global _cam, _cam_tried
    with _cam_lock:
        if _cam_tried:
            return _cam
        _cam_tried = True
        if not config.CAM_PATH.is_file():
            return None
        if not cam_matches_classifier():
            log.warning("%s does not belong to %s (sha mismatch): heatmap off", config.CAM_PATH.name, config.MODEL_PATH.name)
            return None
        try:
            _cam = Cam()
        except Exception:  # noqa: BLE001 - a bad CAM file must not take the kiosk down
            log.exception("CAM model failed to load: heatmap off")
            _cam = None
        return _cam
