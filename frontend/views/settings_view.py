from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from backend import pi_backend_enabled
from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link
from navigation import navigate
from services.storage import data_dir, get_storage


def render_settings_view(*, root: Path) -> None:
    with mobile_frame():
        if render_back_link("Home", key="set_back"):
            navigate("home")
        st.markdown("### Settings")
        store = get_storage()
        mb = store.storage_size_bytes() / (1024 * 1024)
        st.markdown(f"**Storage:** {mb:.1f} MB · `{store.root}`")
        if st.button("Documents cache → Clear cache"):
            store.clear_cache()
            st.success("Cache cleared.")
        st.divider()
        pi_enabled = pi_backend_enabled()
        backend_options = ("local", "mock", "pi") if pi_enabled else ("local", "mock")
        if not pi_enabled and st.session_state.get("inference_backend_kind") == "pi":
            st.session_state["inference_backend_kind"] = "local"
        backend_labels = {
            "local": "PC — browser camera or upload (recommended)",
            "mock": "PC — samples + upload (same model, no camera label)",
            "pi": "Raspberry Pi — Pi camera over LAN",
        }
        st.radio(
            "Backend",
            backend_options,
            format_func=lambda k: backend_labels[k],
            key="inference_backend_kind",
            help="Camera screen always offers webcam + upload on PC. Pi uses the device on the network.",
        )
        if pi_enabled:
            st.text_input("Pi base URL", key="pi_base_url_input")
        if "SKIN_KERAS_PATH_UI" not in st.session_state:
            st.session_state["SKIN_KERAS_PATH_UI"] = os.environ.get(
                "SKIN_KERAS_PATH", str(root / "models" / "skin_classifier_full.keras")
            )
        st.text_input("SKIN_KERAS_PATH", key="SKIN_KERAS_PATH_UI")
        if "pixels_per_mm_ui" not in st.session_state:
            st.session_state["pixels_per_mm_ui"] = 10.0
        st.number_input("pixels_per_mm", 0.1, 100.0, step=0.5, key="pixels_per_mm_ui")
        st.toggle("Block run when quality checks fail", key="strict_quality_gate")
        st.toggle(
            "Enhance image for ABCDE only (color + hair removal)",
            key="preprocess_enabled",
            help="Improves asymmetry/border measurements. The CNN always receives the original photo.",
        )
        st.toggle(
            "Debug: show original vs ABCDE-enhanced on Results",
            key="preprocess_debug",
        )
        kind = st.session_state.get("inference_backend_kind", "mock")
        tta_default = kind != "pi"
        if "tta_toggle" not in st.session_state:
            st.session_state["tta_toggle"] = tta_default
        tta = st.toggle(
            "Test-time augmentation (slower)",
            key="tta_toggle",
            help="Recommended off for Raspberry Pi (latency).",
        )
        os.environ["SKIN_TTA"] = "1" if tta else "0"
        if kind == "pi" and tta:
            st.caption("Pi backend: TTA increases latency on the device.")
        st.caption(f"Data: `{data_dir()}` · Env: DERMASCAN_DATA_DIR, SKIN_MODEL_PATH")
        st.divider()
        if st.text_input('Type DELETE to reset') == "DELETE" and st.button("Reset all data"):
            store.reset_all()
            st.session_state.pop("last_result", None)
            st.success("Reset complete.")
            st.rerun()
