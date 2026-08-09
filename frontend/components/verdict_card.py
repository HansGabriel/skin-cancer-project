"""The verdict — the only risk-language element on a results screen.

Reading order is fixed and deliberate: what to do (headline), why (one muted
line), then the sentence addressed to the person, set in the serif face. That
last line is the only serif on the screen: the machine reports in grotesque,
the advice to a human is typeset like prose.
"""

from __future__ import annotations

import streamlit as st

from services.format import tone_colors
from services.verdict import UIVerdict
from theme.tokens import TOKENS as T


def render_verdict_card(v: UIVerdict) -> None:
    ink, _fill = tone_colors(v.tone)
    reasons = "".join(f'<p class="ds-reason">— {r}</p>' for r in v.reasons)
    st.markdown(
        "<div>"
        f'<p class="ds-verdict-head" style="color:{ink}">{v.headline}</p>'
        f'<div class="ds-verdict-mark" style="background:{ink}"></div>'
        f'<p class="ds-verdict-body">{v.body}</p>'
        f'<p class="ds-advice">{v.advice}</p>'
        f"{reasons}"
        "</div>",
        unsafe_allow_html=True,
    )
