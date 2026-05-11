"""Streamlit-cached resources (import only from Streamlit entrypoints)."""

from __future__ import annotations

import streamlit as st


def _interpreter_class():
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore[import-untyped]

        return tflite.Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-untyped]

            return Interpreter
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install TFLite inference: pip install tflite-runtime "
                "(Pi / older Python) or pip install ai-edge-litert (many Python 3.12+ desktops)."
            ) from exc


@st.cache_resource
def get_tflite_interpreter(model_path: str):
    """Load and allocate TFLite interpreter once per ``model_path`` (rerun-safe)."""
    Interpreter = _interpreter_class()
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter
