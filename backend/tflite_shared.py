"""TFLite interpreter wiring with optional temperature scaling and TTA."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from backend import preprocessing
from backend.contracts import ScanResult
from backend.recommendations import RECOMMENDATIONS

_TEMPERATURE: float | None = None


def load_labels(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _load_temperature() -> float:
    global _TEMPERATURE
    if _TEMPERATURE is not None:
        return _TEMPERATURE
    path = os.environ.get("SKIN_TEMPERATURE_JSON")
    if not path:
        root = Path(__file__).resolve().parent.parent
        path = str(root / "models" / "temperature.json")
    p = Path(path)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        _TEMPERATURE = float(data.get("T", 1.0))
    else:
        _TEMPERATURE = 1.0
    return _TEMPERATURE


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    if _load_temperature() != 1.0:
        x = x / _load_temperature()
    x = x - np.max(x)
    e = np.exp(x)
    return (e / np.sum(e)).astype(np.float32)


def _flip_rgb(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "hflip":
        return np.fliplr(image_rgb)
    if mode == "vflip":
        return np.flipud(image_rgb)
    if mode == "rot180":
        return np.rot90(image_rgb, 2)
    return image_rgb


def run_inference_on_rgb(
    image_rgb: np.ndarray,
    interpreter,
    *,
    use_tta: bool | None = None,
) -> tuple[np.ndarray, int]:
    """Run TFLite; optional 4-view TTA average."""
    if use_tta is None:
        use_tta = os.environ.get("SKIN_TTA", "1") == "1"
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    modes = ("identity", "hflip", "vflip", "rot180") if use_tta else ("identity",)

    probs_acc: np.ndarray | None = None
    t0 = time.perf_counter()
    for mode in modes:
        view = _flip_rgb(image_rgb, mode) if mode != "identity" else image_rgb
        input_tensor = preprocessing.to_input_tensor(view, input_details)
        interpreter.set_tensor(input_details["index"], input_tensor)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details["index"])[0]
        logits = preprocessing.dequantize_output(raw, output_details)
        if logits.max() <= 1.0 and logits.sum() > 0.9:
            p = logits.astype(np.float32)
        else:
            p = _softmax(logits)
        probs_acc = p if probs_acc is None else probs_acc + p
    inference_ms = int((time.perf_counter() - t0) * 1000)
    assert probs_acc is not None
    probs = probs_acc / float(len(modes))
    return probs, inference_ms


def decode_image_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compose_scan_result(
    labels: list[str],
    probs: np.ndarray,
    *,
    image_jpg_bytes: bytes,
    inference_ms: int,
    backend_id: str,
) -> ScanResult:
    idx = int(np.argmax(probs))
    label = labels[idx]
    confidence = float(probs[idx]) * 100.0
    prob_dict = {labels[i]: float(probs[i]) * 100.0 for i in range(len(labels))}
    rec = RECOMMENDATIONS[label]
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    return ScanResult(
        label=label,
        confidence=confidence,
        probs=prob_dict,
        image_jpg_bytes=image_jpg_bytes,
        timestamp=ts,
        inference_ms=inference_ms,
        urgency=rec["urgency"],
        icon=rec["icon"],
        action=rec["action"],
        backend_id=backend_id,
    )
