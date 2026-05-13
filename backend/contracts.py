"""Shared types for inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

BackendKind = Literal["mock", "local", "pi"]


@dataclass(frozen=True)
class ScanResult:
    """Unified scan result for every backend (in-memory image is always raw JPEG bytes)."""

    label: str
    confidence: float
    probs: dict[str, float]
    image_jpg_bytes: bytes
    timestamp: str
    inference_ms: int
    urgency: str
    icon: str
    action: str
    backend_id: str


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol implemented by mock, local TFLite, and Pi HTTP backends."""

    backend_id: str

    def health(self) -> dict:
        """Return a small JSON-serializable status dict for the sidebar."""

    def scan(self, image_jpg_bytes: bytes | None = None) -> ScanResult:
        """Run one scan. Mock/local require bytes. Pi may accept ``image_jpg_bytes`` as multipart ``image`` (debug)."""

    def fetch_log(self) -> tuple[list[dict[str, Any]], str | None]:
        """Return ``(rows, warning)`` for history UI. ``warning`` explains missing /log etc."""
