from __future__ import annotations

import streamlit as st

from theme.tokens import TOKENS as T


def render_empty_state(title: str, message: str) -> None:
    st.markdown(
        f'<div class="ds-empty"><p style="font-size:{T.stat_font}px">{title}</p><p style="font-size:{T.font_xs}px">{message}</p></div>',
        unsafe_allow_html=True,
    )
