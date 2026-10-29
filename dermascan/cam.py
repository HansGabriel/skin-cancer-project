"""Eigen-CAM: "where the scanner looked", for the staff drawer only.

    photo (RGB, <= 1024 px)
      -> resize to 224, same raw [0, 255] input as the classifier
      -> one forward pass, and the last convolution map is read out:
           EfficientNetB0's top_activation, 7 x 7 x 1280 - no probabilities
      -> eigencam_map   first principal component of that map, 7 x 7 in [0, 1]
      -> overlay_jpg    stretched over the photo, JET colours, half-and-half blend

Why Eigen-CAM and not Grad-CAM: the Pi's TFLite runtime has no gradients, so a
heatmap needs a method that works from one forward pass (Muhammad & Yeasin 2020).
It is class-agnostic: it shows what the network found salient, not "why cancer".

Where the map comes from (since 2026-10-29): the deployed classifier itself.
models/skin_classifier.tflite already holds that layer; a second interpreter is
opened on the same file with every intermediate tensor kept
(experimental_preserve_all_tensors), and the 7x7x1280 tensor is read after the
pass. No extra model file, no .keras on the PC, nothing to keep in step with the
classifier. A separate models/skin_classifier_cam.tflite (scripts/export_cam_tflite.py)
is still used first if one is present and its .json names this classifier's sha256 -
kept for the day a quantised export fuses the layer away.

Display only. Nothing here feeds the verdict. About a second on a Pi 4, run on
demand when staff open the drawer, never on the scan path. The interpreter is
built in a background thread after boot (warm_in_background), so the first
drawer is not slow and the boot is not slower.
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

# The layer the map is read from, by name; by shape when the name is not found
# (a converter may rename it). (7, 7) is EfficientNetB0's last map at 224 px.
FEATURE_LAYER_NAME = "top_activation"
FEATURE_MAP_HW = (config.INPUT_SIZE // 32, config.INPUT_SIZE // 32)


def eigencam_map(activations: np.ndarray) -> np.ndarray:
    """(H, W, C) feature map -> (H, W) attention in [0, 1].

    First principal component of the (positions x channels) matrix, sign-corrected
    so it points at the dominant activation mass, rectified and min-max scaled.
    A flat map (nothing salient) gives all zeros.
    """
    h, w, c = activations.shape
    a = activations.reshape(h * w, c).astype(np.float64)
    a -= a.mean(axis=0, keepdims=True)
    with np.errstate(all="ignore"):  # Apple's BLAS warns "divide by zero in matmul" on healthy input
        _u, _s, vt = np.linalg.svd(a, full_matrices=False)
        proj = a @ vt[0]
    if not np.isfinite(proj).all():
        return np.zeros((h, w), dtype=np.float32)
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


def feature_tensor(details: list[dict]) -> dict | None:
    """The tensor to read the map from, among an interpreter's tensor details.

    By name first (top_activation), else the highest-index 4-D tensor with the
    expected 7 x 7 map: tensors are numbered in graph order, so the last such
    tensor is the deepest convolution. None when there is no such tensor - a
    delegate fused it away, or this is not EfficientNetB0.
    """
    maps = [d for d in details if len(d["shape"]) == 4 and tuple(d["shape"][1:3]) == FEATURE_MAP_HW]
    if not maps:
        return None
    named = [d for d in maps if FEATURE_LAYER_NAME in d.get("name", "")]
    return max(named or maps, key=lambda d: d["index"])


def dequantise(tensor: np.ndarray, details: dict) -> np.ndarray:
    """An int8/uint8 tensor back to float, with its own scale and zero point; float stays as is."""
    if tensor.dtype == np.float32:
        return tensor
    scale, zero = details.get("quantization", (0.0, 0))
    if not scale:
        q = details.get("quantization_parameters") or {}
        scales, zeros = q.get("scales", []), q.get("zero_points", [])
        scale, zero = (float(scales[0]) if len(scales) else 1.0), (int(zeros[0]) if len(zeros) else 0)
    return (tensor.astype(np.float32) - float(zero)) * float(scale)


class Cam:
    """One interpreter that can give the feature map; `heatmap` is safe to call from any thread."""

    def __init__(self, interpreter, out_details: dict, source: str) -> None:
        self._it = interpreter
        self._in = interpreter.get_input_details()[0]
        self._out = out_details
        self.source = source  # "cam file" or "classifier", for the log
        self._lock = threading.Lock()

    @classmethod
    def from_cam_file(cls, path: Path = config.CAM_PATH, num_threads: int = config.NUM_THREADS) -> Cam:
        """The separate CAM export: one 4-D output, the map."""
        it = _interpreter_class()(model_path=str(path), num_threads=num_threads)
        it.allocate_tensors()
        outs = [o for o in it.get_output_details() if len(o["shape"]) == 4]
        if not outs:
            raise ValueError(f"{path.name} has no 4-D (feature map) output")
        return cls(it, outs[0], "cam file")

    @classmethod
    def from_classifier(cls, path: Path = config.MODEL_PATH, num_threads: int = config.NUM_THREADS) -> Cam:
        """The deployed classifier, opened a second time with its inner tensors kept."""
        it = _interpreter_class()(model_path=str(path), num_threads=num_threads, experimental_preserve_all_tensors=True)
        it.allocate_tensors()
        out = feature_tensor(it.get_tensor_details())
        if out is None:
            raise ValueError(f"{path.name}: no {FEATURE_MAP_HW[0]}x{FEATURE_MAP_HW[1]} feature map to read")
        return cls(it, out, "classifier")

    def attention(self, rgb: np.ndarray) -> np.ndarray:
        """(7, 7) map in [0, 1] for one photo (single view, no TTA)."""
        small = cv2.resize(rgb, (config.INPUT_SIZE, config.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        x = to_input_tensor(small, self._in)
        with self._lock:
            self._it.set_tensor(self._in["index"], x)
            self._it.invoke()
            feat = dequantise(self._it.get_tensor(self._out["index"]), self._out)[0]
        return eigencam_map(feat)

    def heatmap(self, rgb: np.ndarray) -> bytes:
        """The overlay JPEG for one photo."""
        return overlay_jpg(rgb, self.attention(rgb))


_cam: Cam | None = None
_cam_tried = False
_cam_lock = threading.Lock()


def _build() -> Cam | None:
    """The separate CAM file when it belongs to this classifier, else the classifier itself."""
    if config.CAM_PATH.is_file():
        if cam_matches_classifier():
            return Cam.from_cam_file()
        log.warning("%s does not belong to %s (sha mismatch): reading the map from the classifier instead", config.CAM_PATH.name, config.MODEL_PATH.name)
    cam = Cam.from_classifier()
    # One pass on a blank frame: a delegate that fused the layer away gives an unreadable
    # or all-zero tensor here, and the drawer must not offer a heatmap it cannot draw.
    probe = cam.attention(np.full((config.INPUT_SIZE, config.INPUT_SIZE, 3), 128, np.uint8))
    if probe.shape != FEATURE_MAP_HW:
        raise ValueError(f"feature map came back as {probe.shape}")
    return cam


def get_cam() -> Cam | None:
    """The shared Cam, or None when no heatmap can be built for THIS classifier.

    Tried once per process; a failure is one line in the log and a hidden button,
    never a kiosk that does not start.
    """
    global _cam, _cam_tried
    with _cam_lock:
        if _cam_tried:
            return _cam
        _cam_tried = True
        if not config.MODEL_PATH.is_file() and not config.CAM_PATH.is_file():
            return None
        try:
            _cam = _build()
            log.info("heatmap ready (from the %s)", _cam.source)
        except Exception:  # noqa: BLE001 - a bad CAM file or runtime must not take the kiosk down
            log.exception("heatmap off: could not read a feature map")
            _cam = None
        return _cam


def warm_in_background() -> None:
    """Build the Cam on a thread, so neither the boot nor the first staff tap waits for it."""
    threading.Thread(target=get_cam, name="cam-warm", daemon=True).start()
