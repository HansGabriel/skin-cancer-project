from __future__ import annotations

import streamlit as st


def render_recommendation_card(text: str) -> None:
    st.markdown(f'<div class="ds-rec-card"><p style="margin:0">{text}</p></div>', unsafe_allow_html=True)
