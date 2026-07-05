"""TFLite interpreter wiring with sensitivity-first decision logic and TTA."""

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

_THRESHOLDS: dict | None = None
_TEMPERATURE: float | None = None


def load_labels(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _load_temperature() -> float:
    """Calibration temperature T from models/temperature.json (1.0 => no calibration).

    Fitted on the validation set during training. Used only to make the *displayed*
    confidence honest — NOT to make the cancer-screen decision, which stays on the
    raw probabilities so the validated 90% sensitivity threshold is preserved.
    """
    global _TEMPERATURE
    if _TEMPERATURE is not None:
        return _TEMPERATURE
    path = os.environ.get("SKIN_TEMPERATURE_JSON")
    if not path:
        root = Path(__file__).resolve().parent.parent
        path = str(root / "models" / "temperature.json")
    p = Path(path)
    try:
        T = float(json.loads(p.read_text(encoding="utf-8")).get("T", 1.0)) if p.is_file() else 1.0
    except (ValueError, OSError, KeyError):
        T = 1.0
    # Guard against bad/degenerate values that would distort probabilities.
    _TEMPERATURE = T if T and T > 0 else 1.0
    return _TEMPERATURE


def apply_temperature(probs: np.ndarray, temperature: float | None = None) -> np.ndarray:
    """Re-calibrate softmax probabilities with temperature T.

    The exported TFLite model emits softmax probs (true logits are gone), so we
    recover pseudo-logits via log(p), divide by T, and re-softmax. This is the
    standard post-hoc way to temperature-scale probabilities and is exactly
    softmax(log(p) / T). T>1 softens overconfident predictions toward honesty.
    """
    if temperature is None:
        temperature = _load_temperature()
    if temperature == 1.0:
        return probs.astype(np.float32)
    eps = 1e-12
    pseudo_logits = np.log(np.clip(probs.astype(np.float64), eps, 1.0))
    return _softmax(pseudo_logits / float(temperature))


def _load_thresholds() -> dict:
    """Sensitivity-first decision config from models/thresholds.json (see training Cell 10b).

    Empty dict => fall back to plain argmax (backwards compatible).
    """
    global _THRESHOLDS
    if _THRESHOLDS is not None:
        return _THRESHOLDS
    path = os.environ.get("SKIN_THRESHOLDS_JSON")
    if not path:
        root = Path(__file__).resolve().parent.parent
        path = str(root / "models" / "thresholds.json")
    p = Path(path)
    _THRESHOLDS = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return _THRESHOLDS


def decide_index(probs: np.ndarray) -> int:
    """Pick the class index. If thresholds.json defines a cancer-screen threshold, flag the
    case as cancerous when p(pre_cancerous)+p(malignant) clears it (sensitivity-first), then
    report whichever cancer class is more probable; otherwise pick the best non-cancer class.
    Falls back to argmax when no threshold config is present.
    """
    th = _load_thresholds()
    thr = th.get("screen_cancer_threshold")
    pre_i = th.get("precancer_idx")
    mal_i = th.get("malignant_idx")
    if thr is None or pre_i is None or mal_i is None:
        return int(np.argmax(probs))
    if max(pre_i, mal_i) >= len(probs):
        return int(np.argmax(probs))
    p_cancer = float(probs[pre_i]) + float(probs[mal_i])
    if p_cancer >= float(thr):
        return mal_i if probs[mal_i] >= probs[pre_i] else pre_i
    masked = [p if i not in (pre_i, mal_i) else -1.0 for i, p in enumerate(probs)]
    return int(np.argmax(masked))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Fallback softmax for models that output raw logits (not softmax)."""
    x = logits.astype(np.float64) - np.max(logits)
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
    # Decide the class on the RAW probabilities so the validated, sensitivity-first
    # cancer-screen threshold (tuned on raw softmax, training Cell 12) is preserved.
    idx = decide_index(probs)
    label = labels[idx]
    # Report confidence / per-class probabilities on the TEMPERATURE-CALIBRATED
    # probabilities so the numbers shown to the user aren't overconfident.
    cal = apply_temperature(probs)
    confidence = float(cal[idx]) * 100.0
    prob_dict = {labels[i]: float(cal[i]) * 100.0 for i in range(len(labels))}
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
