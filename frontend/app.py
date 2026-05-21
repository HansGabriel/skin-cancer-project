"""DermaScan v2 — screen router entry."""

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
from navigation import current_route, init_navigation, navigate
from services.auth import enforce_passcode_gate
from services.inference import get_inference_backend
from theme.css import inject_global_css
from views.camera_view import render_camera_view
from views.case_view import render_case_view
from views.folder_view import render_folder_view
from views.history_view import render_history_view
from views.home_view import render_home_view
from views.results_view import render_results_view
from views.settings_view import render_settings_view


def _model_path() -> str:
    return os.environ.get("SKIN_MODEL_PATH", str(ROOT / "models" / "skin_classifier.tflite"))


def _labels_path() -> str:
    return os.environ.get("SKIN_LABELS_PATH", str(ROOT / "models" / "labels.txt"))


def _default_pi_url() -> str:
    return os.environ.get("PI_BASE_URL", "http://raspberrypi.local:5000")


def _init_session() -> None:
    init_navigation()
    if "SKIN_KERAS_PATH_UI" not in st.session_state:
        st.session_state["SKIN_KERAS_PATH_UI"] = os.environ.get(
            "SKIN_KERAS_PATH", str(ROOT / "models" / "skin_classifier_full.keras")
        )
    st.session_state.setdefault("pixels_per_mm_ui", 10.0)
    st.session_state.setdefault("strict_quality_gate", True)
    st.session_state.setdefault("inference_backend_kind", "local")
    st.session_state.setdefault("pi_base_url_input", _default_pi_url())
    st.session_state.setdefault("preprocess_enabled", True)
    st.session_state.setdefault("preprocess_debug", False)
    if st.session_state["inference_backend_kind"] == "pi":
        st.session_state.setdefault("tta_toggle", False)
        os.environ.setdefault("SKIN_TTA", "0")
    else:
        st.session_state.setdefault("tta_toggle", True)
        os.environ.setdefault("SKIN_TTA", "1")


def main() -> None:
    st.set_page_config(
        page_title="DermaScan AI",
        page_icon="🧬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_global_css()
    enforce_passcode_gate()
    _init_session()

    kind = cast(BackendKind, st.session_state["inference_backend_kind"])
    backend = get_inference_backend(
        kind, _model_path(), _labels_path(), str(st.session_state["pi_base_url_input"]).rstrip("/")
    )

    with st.sidebar:
        st.caption("Power user")
        if st.button("Settings", key="side_settings"):
            navigate("settings")
        if st.button("Health check", key="health_btn"):
            st.write(backend.health())

    route = current_route()
    if route == "home":
        render_home_view()
    elif route == "camera":
        render_camera_view(root=ROOT, backend=backend, kind=kind)
    elif route == "results":
        render_results_view(root=ROOT, model_path=_model_path())
    elif route == "history":
        render_history_view()
    elif route == "folder":
        render_folder_view()
    elif route == "case":
        render_case_view()
    elif route == "settings":
        render_settings_view(root=ROOT)
    else:
        navigate("home")


if __name__ == "__main__":
    main()
