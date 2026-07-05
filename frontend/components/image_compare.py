from __future__ import annotations

import streamlit as st


def render_image_compare(rgb, overlay_jpg: bytes | None, *, gradcam_caption: str | None = None) -> None:
    """Photo + optional AI-attention overlay.

    When no overlay is available, show only the photo — the technical reason
    (Grad-CAM/keras path) belongs in the Technical details expander, not here.
    ``gradcam_caption`` is kept for backwards compatibility but no longer shown
    to end users.
    """
    if not overlay_jpg:
        if rgb is not None:
            st.caption("Your photo")
            st.image(rgb, use_container_width=True)
        return
    left, right = st.columns(2)
    with left:
        st.caption("Your photo")
        if rgb is not None:
            st.image(rgb, use_container_width=True)
    with right:
        st.caption("Where the AI looked")
        st.image(overlay_jpg, use_container_width=True)
