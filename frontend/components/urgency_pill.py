"""Compact verdict pill for list rows and secondary screens.

Renders the SAME wording as the results screen. It takes a ``UIVerdict`` rather
than a risk band on purpose: deriving separate copy from the composite score is
what let a scan read "URGENT" in one place and "NOT SURE" in another.
"""

from __future__ import annotations

import streamlit as st

from services.format import tone_colors
from services.verdict import UIVerdict
from theme.tokens import TOKENS as T


def render_verdict_pill(v: UIVerdict) -> None:
    ink, fill = tone_colors(v.tone)
    st.markdown(
        f'<span style="display:inline-block;padding:6px 14px;border-radius:999px;'
        f"background:{fill};color:{ink};border:1px solid {ink}22;"
        f'font-weight:700;font-size:{T.pill_font}px;letter-spacing:.03em">{v.headline}</span>',
        unsafe_allow_html=True,
    )
