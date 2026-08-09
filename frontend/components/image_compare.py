from __future__ import annotations

import streamlit as st

from theme.tokens import TOKENS as T


def render_image_compare(rgb, overlay_jpg: bytes | None, *, note: str | None = None) -> None:
    """Photo + optional attention overlay, with the note that explains the overlay.

    The note must render whenever the overlay does. Showing a heatmap of "where
    the AI looked" with no explanation invites people to read it as the scanner
    outlining a lesion, which is exactly what it is not.
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
        st.caption("Where the scanner looked")
        st.image(overlay_jpg, use_container_width=True)
    if note:
        st.markdown(
            f'<p style="color:{T.text_muted};font-size:{T.font_xs}px;margin:4px 0 0">{note}</p>',
            unsafe_allow_html=True,
        )
