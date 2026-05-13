"""Grad-CAM toggle + images."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import streamlit as st


def render_gradcam_viewer(
    rgb: np.ndarray | None,
    overlay_jpg: bytes | None,
    disclaimer: str | None,
) -> None:
    st.subheader("Explainability (Grad-CAM++)")
    if rgb is None:
        st.warning("No image array for visualization.")
        return
    if overlay_jpg is None:
        st.image(rgb, caption="Original capture", width="stretch")
        st.caption("Grad-CAM needs a compatible `SKIN_KERAS_PATH` EfficientNet model on this PC.")
        if disclaimer:
            st.caption(disclaimer)
        return

    mode = st.radio(
        "View",
        ("Original", "Heatmap overlay", "Side by side"),
        horizontal=True,
        key="gradcam_view_mode",
    )
    if mode == "Original":
        st.image(rgb, caption="Original", width="stretch")
    elif mode == "Heatmap overlay":
        st.image(BytesIO(overlay_jpg), caption="Grad-CAM overlay", width="stretch")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.image(rgb, caption="Original", width="stretch")
        with c2:
            st.image(BytesIO(overlay_jpg), caption="Grad-CAM overlay", width="stretch")
    if disclaimer:
        st.caption(disclaimer)
