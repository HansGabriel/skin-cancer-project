from __future__ import annotations

import streamlit as st


def render_empty_state(title: str, message: str) -> None:
    st.markdown(
        f'<div class="ds-empty"><p style="font-size:18px">{title}</p><p style="font-size:13px">{message}</p></div>',
        unsafe_allow_html=True,
    )
