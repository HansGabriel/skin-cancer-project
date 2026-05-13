"""Streamlit entry: tabbed DermaScan v2."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
from pathlib import Path
from typing import cast

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRONTEND))

from backend.contracts import BackendKind
from services.inference import get_inference_backend
from views.case_history_view import render_case_history_tab
from views.results_view import render_results_tab
from views.scan_view import render_scan_tab
from views.settings_view import render_settings_tab


def _model_path() -> str:
    return os.environ.get("SKIN_MODEL_PATH", str(ROOT / "models" / "skin_classifier.tflite"))


def _labels_path() -> str:
    return os.environ.get("SKIN_LABELS_PATH", str(ROOT / "models" / "labels.txt"))


def _default_pi_url() -> str:
    return os.environ.get("PI_BASE_URL", "http://raspberrypi.local:5000")


def main() -> None:
    st.set_page_config(page_title="DermaScan AI", page_icon="🧬", layout="wide")
    st.title("🧬 DermaScan AI — Skin lesion risk screening")
    st.caption("AI-assisted dermatoscope prototype. For screening only.")

    default_keras = os.environ.get(
        "SKIN_KERAS_PATH",
        str(ROOT / "models" / "skin_classifier_full.keras"),
    )
    if "SKIN_KERAS_PATH_UI" not in st.session_state:
        st.session_state["SKIN_KERAS_PATH_UI"] = default_keras
    if "pixels_per_mm_ui" not in st.session_state:
        st.session_state["pixels_per_mm_ui"] = 10.0
    if "strict_quality_gate" not in st.session_state:
        st.session_state["strict_quality_gate"] = True

    with st.sidebar:
        st.subheader("Inference backend")
        kind = cast(
            BackendKind,
            st.radio(
                "Source",
                ("mock", "local", "pi"),
                format_func=lambda k: {
                    "mock": "Upload / sample images (no camera)",
                    "local": "Browser camera (this machine)",
                    "pi": "Raspberry Pi (HTTP)",
                }[k],
                key="inference_backend_kind",
            ),
        )
        pi_base = st.text_input("Pi base URL", value=_default_pi_url(), key="pi_base_url_input")
        st.caption("Env: `SKIN_MODEL_PATH`, `SKIN_LABELS_PATH`, `PI_BASE_URL`.")

        backend = get_inference_backend(
            kind,
            _model_path(),
            _labels_path(),
            pi_base.rstrip("/"),
        )

        if st.button("Check backend health", key="health_btn"):
            status = backend.health()
            if status.get("status") == "ok":
                st.success(status)
            else:
                st.error(status)

    tab_scan, tab_results, tab_cases, tab_settings = st.tabs(
        ("📸 Scan", "📊 Results", "📁 Cases", "⚙️ Settings")
    )

    with tab_scan:
        render_scan_tab(root=ROOT, backend=backend, kind=kind)

    with tab_results:
        render_results_tab(backend=backend)

    with tab_cases:
        render_case_history_tab()

    with tab_settings:
        render_settings_tab(root=ROOT)


if __name__ == "__main__":
    main()
