"""Case row in history tree."""

from __future__ import annotations

import streamlit as st

from services.format import urgency_from_band
from services.storage import Case
from theme.tokens import TOKENS as T


def render_case_row(case: Case, *, urgency: str, date: str, key: str) -> bool:
    """Tree row: case name, urgency chip, date."""
    label, color = urgency_from_band(urgency)
    chip = (
        f'<span style="font-size:10px;padding:2px 8px;border-radius:999px;'
        f'background:{color};color:#111;font-weight:700">{label}</span>'
    )
    st.markdown(
        f'<div class="ds-case-row"><span style="font-weight:600">{case.name}</span> '
        f'{chip} <span style="color:{T.text_muted};font-size:12px">{date}</span></div>',
        unsafe_allow_html=True,
    )
    return st.button(f"Open {case.name}", key=key, use_container_width=True)
