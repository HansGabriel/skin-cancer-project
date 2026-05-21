from __future__ import annotations

import streamlit as st

from theme.tokens import TOKENS as T


def render_detection_headline(label: str, confidence: float) -> None:
    display = label.upper().replace("_", " ")
    st.markdown(
        f'<p style="font-size:{T.type_sm}px;color:{T.text_muted};margin:0">Detection</p>'
        f'<p style="font-size:{T.type_2xl}px;font-weight:700">{display}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span style="padding:4px 10px;border-radius:999px;background:{T.success};'
        f'color:#111;font-size:13px;font-weight:600">↑ {confidence:.1f}% Confidence</span>',
        unsafe_allow_html=True,
    )
