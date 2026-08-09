"""Case row in history tree."""

from __future__ import annotations

import html

import streamlit as st

from services.format import tone_colors
from services.storage import Case
from services.verdict import UIVerdict
from theme.tokens import TOKENS as T


def render_case_row(case: Case, *, verdict: UIVerdict | None, date: str, key: str) -> bool:
    """Tree row: case name, verdict chip, date.

    Takes the verdict rather than a risk band so the chip repeats exactly what
    the result screen said about this scan.
    """
    if verdict is None:
        chip = (
            f'<span style="font-size:{T.chip_font}px;color:{T.text_muted}">No scans yet</span>'
        )
    else:
        ink, fill = tone_colors(verdict.tone)
        chip = (
            f'<span style="font-size:{T.chip_font}px;padding:2px 8px;border-radius:999px;'
            f'background:{fill};color:{ink};font-weight:700">{verdict.headline}</span>'
        )
    st.markdown(
        f'<div class="ds-case-row"><span style="font-weight:600">{html.escape(case.name)}</span> '
        f'{chip} <span style="color:{T.text_muted};font-size:{T.pill_font}px">{date}</span></div>',
        unsafe_allow_html=True,
    )
    return st.button(f"Open {html.escape(case.name)}", key=key, use_container_width=True)
