from __future__ import annotations

import streamlit as st

from components.icons import icon
from theme.tokens import TOKENS as T

_ACCENT = {
    "high": (T.urgent, T.urgent_tint, "alert"),
    "moderate": (T.warning, T.warning_tint, "info"),
    "low": (T.success, T.success_tint, "info"),
}


def render_recommendation_card(text: str, band: str = "low") -> None:
    color, tint, glyph = _ACCENT.get(band, (T.violet, T.violet_tint, "info"))
    st.markdown(
        f'<div class="ds-advice" style="background:{tint};border-left-color:{color}">'
        f'<span style="color:{color}">{icon(glyph, size=20)}</span>'
        f'<div><div style="font-weight:700;font-size:13px;color:{color}">Recommendation</div>'
        f'<p style="margin:2px 0 0">{text}</p></div></div>',
        unsafe_allow_html=True,
    )
