from __future__ import annotations

import streamlit as st

from theme.tokens import TOKENS as T

_BAND = {
    "high": ("URGENT", T.urgent, "#fff"),
    "moderate": ("IMPORTANT", T.warning, "#111"),
    "low": ("LOW CONCERN", T.success, "#111"),
}


def render_urgency_from_band(band: str) -> None:
    label, bg, fg = _BAND.get(band, ("SCREENING", T.info, "#fff"))
    st.markdown(
        f'<span style="display:inline-block;padding:6px 14px;border-radius:999px;background:{bg};'
        f'color:{fg};font-weight:700;font-size:12px;">{label}</span>',
        unsafe_allow_html=True,
    )
