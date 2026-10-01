"""The TFLite skin-lesion classifier, and the rule that turns its output into a label.

What it does, in order:

1. Resize the photo to 224x224 (the size the model was trained at).
2. Run the model on four views of it (as-is, mirrored, upside down, rotated
   180) and average the three probabilities. This is "test-time augmentation";
   it steadies the answer at the cost of three extra runs.
3. Decide the label with the screening rule in models/thresholds.json:
   if p(pre_cancerous) + p(malignant) >= threshold, the spot is FLAGGED and the
   label is whichever of those two is larger; otherwise it is "benign". The
   threshold (about 0.1) is chosen by training/calibrate.py so that 90% of
   cancers in the validation set are flagged. It is deliberately low: missing
   a cancer costs more than a false alarm.
4. Report confidence from TEMPERATURE-SCALED probabilities. The raw softmax is
   over-confident; dividing its log by T (models/temperature.json, fitted on
   validation data) makes the displayed number honest. The decision in step 3
   always uses the raw probabilities, so calibration can never change a label.

One Classifier is shared by every request. A TFLite interpreter is not safe to
use from two threads at once, so `predict` holds a lock while it runs.

The model expects float32 RGB in [0, 255] - no scaling. EfficientNet rescales
internally. Do not add "/255" or "/127.5 - 1".
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dermascan import config

FLAGGED_LABELS = ("pre_cancerous", "malignant")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence_pct: float  # calibrated probability of `label`, 0-100
    probs_pct: dict[str, float]  # calibrated, 0-100, one per label
    raw_probs: tuple[float, ...]  # what the model actually said, 0-1
    inference_ms: int
    flagged: bool  # True when the screening rule flagged the spot

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence_pct": round(self.confidence_pct, 1),
            "probs_pct": {k: round(v, 1) for k, v in self.probs_pct.items()},
            "inference_ms": self.inference_ms,
            "flagged": self.flagged,
        }


# --- Loading --------------------------------------------------------------------


def _interpreter_class():
    """ai-edge-litert on Python 3.12+ (Pi and Mac); tflite-runtime on older setups."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-untyped]

        return Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[import-untyped]

            return Interpreter
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install the model runtime: pip install ai-edge-litert") from exc


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_thresholds(path: Path) -> dict:
    """Empty dict when the file is absent: the decision falls back to argmax."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_temperature(path: Path) -> float:
    """1.0 (no change) when the file is absent or holds a bad value."""
    try:
        t = float(json.loads(path.read_text(encoding="utf-8")).get("T", 1.0)) if path.is_file() else 1.0
    except (ValueError, OSError, KeyError):
        t = 1.0
    return t if t > 0 else 1.0


# --- The maths, as plain functions so the tests can hit them directly --------------


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64) - np.max(logits)
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log(p) / T). T > 1 softens over-confident probabilities."""
    if temperature == 1.0:
        return probs.astype(np.float32)
    pseudo_logits = np.log(np.clip(probs.astype(np.float64), 1e-12, 1.0))
    return softmax(pseudo_logits / float(temperature))


def decide_index(probs: np.ndarray, thresholds: dict) -> int:
    """The screening rule. See the module docstring, step 3."""
    thr = thresholds.get("screen_cancer_threshold")
    pre_i = thresholds.get("precancer_idx")
    mal_i = thresholds.get("malignant_idx")
    if thr is None or pre_i is None or mal_i is None or max(pre_i, mal_i) >= len(probs):
        return int(np.argmax(probs))
    if float(probs[pre_i]) + float(probs[mal_i]) >= float(thr):
        return mal_i if probs[mal_i] >= probs[pre_i] else pre_i
    not_cancer = [p if i not in (pre_i, mal_i) else -1.0 for i, p in enumerate(probs)]
    return int(np.argmax(not_cancer))


def to_input_tensor(rgb_224: np.ndarray, input_details: dict) -> np.ndarray:
    """Float [0, 255] as a (1, 224, 224, 3) batch; quantised only if the model asks."""
    image = np.expand_dims(rgb_224.astype(np.float32), axis=0)
    dtype = input_details["dtype"]
    scale, zero_point = input_details["quantization"]
    if dtype == np.float32:
        return image
    if scale == 0:
        raise ValueError("Quantised input with scale 0")
    q = np.round(image / scale + zero_point)
    if dtype == np.uint8:
        return np.clip(q, 0, 255).astype(np.uint8)
    if dtype == np.int8:
        return np.clip(q, -128, 127).astype(np.int8)
    raise TypeError(f"Unsupported model input type: {dtype}")


def dequantize_output(raw: np.ndarray, output_details: dict) -> np.ndarray:
    if output_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_details["quantization"]
        if scale == 0:
            raise ValueError("Quantised output with scale 0")
        return (raw.astype(np.float32) - zero_point) * scale
    return raw.astype(np.float32)


def _views(rgb_224: np.ndarray, tta: bool) -> list[np.ndarray]:
    if not tta:
        return [rgb_224]
    return [rgb_224, np.fliplr(rgb_224), np.flipud(rgb_224), np.rot90(rgb_224, 2)]


# --- The classifier object ----------------------------------------------------


class Classifier:
    """Owns one loaded model. Create it once; `predict` as often as you like."""

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        labels_path: Path = config.LABELS_PATH,
        thresholds_path: Path = config.THRESHOLDS_PATH,
        temperature_path: Path = config.TEMPERATURE_PATH,
        *,
        num_threads: int = config.NUM_THREADS,
        use_tta: bool = config.USE_TTA,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = load_labels(Path(labels_path))
        self.thresholds = load_thresholds(Path(thresholds_path))
        self.temperature = load_temperature(Path(temperature_path))
        self.use_tta = use_tta
        self._interpreter = _interpreter_class()(model_path=str(self.model_path), num_threads=num_threads)
        self._interpreter.allocate_tensors()
        self._lock = threading.Lock()
        self._in = self._interpreter.get_input_details()[0]
        self._out = self._interpreter.get_output_details()[0]
        if len(self.labels) != int(self._out["shape"][-1]):
            raise ValueError(
                f"{labels_path} has {len(self.labels)} labels but the model outputs {self._out['shape'][-1]}"
            )

    def predict(self, rgb: np.ndarray) -> Prediction:
        """`rgb` is any HxWx3 uint8 RGB image. Returns the label and honest confidence."""
        rgb_224 = cv2.resize(rgb, (config.INPUT_SIZE, config.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        views = _views(rgb_224, self.use_tta)
        total = np.zeros(len(self.labels), dtype=np.float32)
        with self._lock:
            t0 = time.perf_counter()
            for view in views:
                self._interpreter.set_tensor(self._in["index"], to_input_tensor(np.ascontiguousarray(view), self._in))
                self._interpreter.invoke()
                # The model ends in a softmax layer, so this is already probabilities.
                total += dequantize_output(self._interpreter.get_tensor(self._out["index"])[0], self._out)
            inference_ms = int((time.perf_counter() - t0) * 1000)
        raw = total / float(total.sum())  # average of the views; renormalised against rounding

        idx = decide_index(raw, self.thresholds)
        calibrated = apply_temperature(raw, self.temperature)
        label = self.labels[idx]
        return Prediction(
            label=label,
            confidence_pct=float(calibrated[idx]) * 100.0,
            probs_pct={self.labels[i]: float(calibrated[i]) * 100.0 for i in range(len(self.labels))},
            raw_probs=tuple(float(p) for p in raw),
            inference_ms=inference_ms,
            flagged=label in FLAGGED_LABELS,
        )


_SHARED: Classifier | None = None
_SHARED_LOCK = threading.Lock()


def get_classifier() -> Classifier:
    """The one classifier the app shares. Loading takes ~1 s on a Pi, so do it once."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = Classifier()
        return _SHARED
