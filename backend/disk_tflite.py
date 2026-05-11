"""Shared TFLite-on-disk logic for mock and local (browser) backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend import scan_log
from backend.contracts import ScanResult
from backend.streamlit_resources import get_tflite_interpreter
from backend.tflite_shared import (
    compose_scan_result,
    decode_image_bytes_to_rgb,
    load_labels,
    run_inference_on_rgb,
)


class DiskTfliteBackend:
    """Load ``.tflite`` from disk; classify JPEG bytes. Used by mock and local backends."""

    def __init__(self, model_path: str, labels_path: str, backend_id: str) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.backend_id = backend_id
        self._labels = load_labels(labels_path)

    def health(self) -> dict:
        mp = Path(self.model_path)
        if not mp.is_file():
            return {"status": "error", "reason": f"Missing model file: {self.model_path}"}
        lp = Path(self.labels_path)
        if not lp.is_file():
            return {"status": "error", "reason": f"Missing labels file: {self.labels_path}"}
        try:
            get_tflite_interpreter(str(mp.resolve()))
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": repr(exc)}
        return {"status": "ok", "model": str(mp.resolve()), "labels": str(lp.resolve())}

    def scan(self, image_jpg_bytes: bytes | None = None) -> ScanResult:
        if not image_jpg_bytes:
            raise ValueError("Provide image bytes (upload, sample image, or camera capture).")
        interpreter = get_tflite_interpreter(str(Path(self.model_path).resolve()))
        rgb = decode_image_bytes_to_rgb(image_jpg_bytes)
        probs, inference_ms = run_inference_on_rgb(rgb, interpreter)
        result = compose_scan_result(
            self._labels,
            probs,
            image_jpg_bytes=image_jpg_bytes,
            inference_ms=inference_ms,
            backend_id=self.backend_id,
        )
        scan_log.insert_scan(
            backend_id=self.backend_id,
            label=result.label,
            confidence=result.confidence,
            probs=result.probs,
            inference_ms=result.inference_ms,
            ts_iso=result.timestamp,
        )
        return result

    def fetch_log(self) -> tuple[list[dict[str, Any]], str | None]:
        rows = scan_log.fetch_recent()
        return rows, None
