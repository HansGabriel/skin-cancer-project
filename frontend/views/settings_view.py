"""Settings: Keras path + calibration."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st


def render_settings_tab(root: Path) -> None:
    st.subheader("Settings")
    default_keras = os.environ.get(
        "SKIN_KERAS_PATH",
        str(root / "models" / "skin_classifier_full.keras"),
    )
    if "SKIN_KERAS_PATH_UI" not in st.session_state:
        st.session_state["SKIN_KERAS_PATH_UI"] = default_keras

    st.text_input(
        "SKIN_KERAS_PATH (Keras for Grad-CAM; 7 outputs for differential)",
        key="SKIN_KERAS_PATH_UI",
    )

    if "pixels_per_mm_ui" not in st.session_state:
        st.session_state["pixels_per_mm_ui"] = 10.0
    st.number_input(
        "pixels_per_mm (demo calibration for diameter)",
        min_value=0.1,
        max_value=100.0,
        step=0.5,
        key="pixels_per_mm_ui",
    )
    st.caption("Tune with a ruler at a fixed working distance; default is illustrative only.")

    st.toggle(
        "Block run when quality checks fail",
        help="When off, the app still runs TFLite + ABCDE but shows quality warnings on the Results tab.",
        key="strict_quality_gate",
    )

    st.markdown("**Env:** `SKIN_MODEL_PATH`, `SKIN_LABELS_PATH`, `PI_BASE_URL`, `SKIN_KERAS_PATH`.")
