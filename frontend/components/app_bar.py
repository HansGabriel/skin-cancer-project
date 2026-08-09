"""The top bar: who this is, what it is not, and the time.

Navigation deliberately lives only in the bottom bar. This bar used to also
carry two icon buttons rendered as ``st.button("")`` — an SVG glyph floating
above an empty button shell, which is what those two blank boxes on the home
screen were.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from services.kiosk import is_kiosk
from theme.tokens import TOKENS as T


def format_app_bar_time(now: datetime | None = None) -> str:
    dt = now or datetime.now()
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{dt.minute:02d}{ampm} {dt.strftime('%a, %b')} {dt.day}, {dt.year}"


def render_app_bar(*, show_home: bool = False) -> None:  # noqa: ARG001 — nav is the bottom bar now
    offline = '<span class="ds-offline-pill">OFFLINE</span>' if is_kiosk() else ""
    st.markdown(
        '<div style="display:flex;align-items:flex-end;justify-content:space-between;'
        f'gap:{T.space_16}px;padding:{T.space_8}px 0 {T.space_12}px;'
        f'border-bottom:1px solid {T.outline};margin-bottom:{T.space_16}px">'
        "<div>"
        f'<div class="ds-brand"><span class="ds-brand-dot"></span>{T.brand_name}{offline}</div>'
        f'<div class="ds-brand-sub">{T.brand_tagline}</div>'
        "</div>"
        f'<div class="ds-app-bar-time">{format_app_bar_time()}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_disclaimer_footer() -> None:
    st.markdown(
        f'<p style="margin:{T.space_16}px 0 0;padding-top:{T.space_12}px;'
        f'border-top:1px solid {T.outline};font-size:{T.font_xs}px;color:{T.text_muted};'
        'line-height:1.5"><strong style="color:inherit">Not a diagnosis.</strong> '
        "This is an educational screening aid. Always contact a qualified health "
        "professional.</p>",
        unsafe_allow_html=True,
    )
