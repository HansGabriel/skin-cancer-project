"""Thin Streamlit-facing alias for cached TFLite / Pi backends."""

from __future__ import annotations

from backend.contracts import InferenceBackend
from backend.streamlit_resources import get_cached_backend


def get_inference_backend(
    kind: str,
    model_path: str,
    labels_path: str,
    pi_base_url: str,
) -> InferenceBackend:
    return get_cached_backend(kind, model_path, labels_path, pi_base_url)
