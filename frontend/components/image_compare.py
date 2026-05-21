from __future__ import annotations

import streamlit as st


def render_image_compare(rgb, overlay_jpg: bytes | None, *, gradcam_caption: str | None = None) -> None:
    left, right = st.columns(2)
    with left:
        st.caption("Original")
        if rgb is not None:
            st.image(rgb, use_container_width=True)
    with right:
        st.caption("Grad-CAM overlay")
        if overlay_jpg:
            st.image(overlay_jpg, use_container_width=True)
        elif rgb is not None:
            st.image(rgb, use_container_width=True)
            if gradcam_caption:
                st.caption(gradcam_caption)
