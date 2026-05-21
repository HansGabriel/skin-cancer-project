"""Swappable inference backends for the Streamlit skin-lesion demo."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from backend.contracts import BackendKind

if TYPE_CHECKING:  # pragma: no cover
    from backend.contracts import InferenceBackend


def pi_backend_enabled() -> bool:
    return os.environ.get("ENABLE_PI_BACKEND") == "1"


def get_backend(
    kind: BackendKind,
    *,
    model_path: str,
    labels_path: str,
    pi_base_url: str = "http://raspberrypi.local:5000",
) -> "InferenceBackend":
    if kind == "mock":
        from backend.mock import MockTfliteBackend

        return MockTfliteBackend(model_path=model_path, labels_path=labels_path)
    if kind == "local":
        from backend.local import LocalTfliteBackend

        return LocalTfliteBackend(model_path=model_path, labels_path=labels_path)
    if kind == "pi":
        if not pi_backend_enabled():
            raise RuntimeError(
                "Pi backend disabled on this deployment. Set ENABLE_PI_BACKEND=1 to enable."
            )
        from backend.pi_http import PiHttpBackend

        return PiHttpBackend(base_url=pi_base_url.rstrip("/"))
    raise ValueError(f"Unknown backend kind: {kind!r}")
