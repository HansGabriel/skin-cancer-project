"""TFLite interpreter wiring (delegates tensor math to ``preprocessing``)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from backend import preprocessing
from backend.contracts import ScanResult
from backend.recommendations import RECOMMENDATIONS


def load_labels(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_inference_on_rgb(image_rgb: np.ndarray, interpreter) -> tuple[np.ndarray, int]:
    """Run TFLite on a single HxWx3 RGB uint8 image. Returns (probabilities float vector, inference_ms)."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_tensor = preprocessing.to_input_tensor(image_rgb, input_details)
    t0 = time.perf_counter()
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details["index"])[0]
    inference_ms = int((time.perf_counter() - t0) * 1000)

    probs = preprocessing.dequantize_output(raw, output_details)
    return probs, inference_ms


def decode_image_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG (etc.) bytes to RGB uint8 array via OpenCV."""
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
    """Build ``ScanResult`` from dequantized probability vector (same order as ``labels``)."""
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
