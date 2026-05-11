"""Mock backend: classify uploaded or bundled sample JPEGs (no camera, no Pi)."""

from __future__ import annotations

from backend.disk_tflite import DiskTfliteBackend


class MockTfliteBackend(DiskTfliteBackend):
    def __init__(self, model_path: str, labels_path: str) -> None:
        super().__init__(model_path, labels_path, backend_id="mock")
