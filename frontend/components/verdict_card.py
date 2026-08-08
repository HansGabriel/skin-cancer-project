"""The single verdict card — the only risk-language element on Results."""

from __future__ import annotations

import streamlit as st

from services.verdict import UIVerdict
from theme.tokens import TOKENS as T

_TONE = {
    "success": (T.success, T.success_tint),
    "warning": (T.warning, T.warning_tint),
    "urgent": (T.urgent, T.urgent_tint),
    "info": (T.info, T.info_tint),
}


def render_verdict_card(v: UIVerdict) -> None:
    color, tint = _TONE.get(v.tone, (T.info, T.info_tint))
    reasons = "".join(
        f'<div style="color:{T.text_muted};font-size:{T.font_sm}px;'
        f'margin-top:{T.space_4}px">• {r}</div>'
        for r in v.reasons
    )
    st.markdown(
        f'<div style="background:{tint};border:1px solid {T.outline};'
        f"border-radius:{T.radius_md}px;padding:{T.space_20}px;"
        f'margin-bottom:{T.space_12}px">'
        f'<div style="color:{color};font-weight:800;font-size:{T.font_xl}px;'
        f'letter-spacing:.04em">{v.headline}</div>'
        f'<div style="color:{T.text};font-size:{T.font_base}px;'
        f'margin-top:{T.space_8}px">{v.body}</div>'
        f'<div style="color:{T.text};font-size:{T.font_base}px;font-weight:600;'
        f'margin-top:{T.space_8}px">{v.advice}</div>'
        f"{reasons}</div>",
        unsafe_allow_html=True,
    )
