"""Local backend: classify JPEG from browser camera (``st.camera_input``)."""

from __future__ import annotations

from backend.disk_tflite import DiskTfliteBackend


class LocalTfliteBackend(DiskTfliteBackend):
    def __init__(self, model_path: str, labels_path: str) -> None:
        super().__init__(model_path, labels_path, backend_id="local")
