from __future__ import annotations

from datetime import datetime

import streamlit as st

from components.icons import icon
from navigation import navigate
from theme.tokens import TOKENS as T


def format_app_bar_time(now: datetime | None = None) -> str:
    dt = now or datetime.now()
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{dt.minute:02d}{ampm} {dt.strftime('%a, %b')} {dt.day}, {dt.year}"


def _icon_nav(col, name: str, key: str, route: str) -> None:
    """Render an SVG icon above a slim click target in the given column."""
    with col:
        st.markdown(f'<div class="ds-iconbtn">{icon(name)}</div>', unsafe_allow_html=True)
        if st.button("", key=key, help=route.capitalize(), use_container_width=True):
            navigate(route)


def render_app_bar(*, show_home: bool = False) -> None:
    # Left: brand wordmark. Right: SVG icon nav buttons.
    n_icons = 3 if show_home else 2
    weights = [5] + [1] * n_icons
    cols = st.columns(weights)
    with cols[0]:
        st.markdown(
            '<div class="ds-brand"><span class="ds-brand-dot"></span>DermaScan</div>'
            '<div class="ds-brand-sub">Skin check · educational</div>',
            unsafe_allow_html=True,
        )
    idx = 1
    if show_home:
        _icon_nav(cols[idx], "home", "app_bar_home", "home")
        idx += 1
    _icon_nav(cols[idx], "folder", "app_bar_history", "history")
    _icon_nav(cols[idx + 1], "gear", "app_bar_settings", "settings")
    st.markdown(f'<p class="ds-app-bar-time">{format_app_bar_time()}</p>', unsafe_allow_html=True)


def render_disclaimer_footer() -> None:
    st.markdown(
        f'<div class="ds-advice" style="background:transparent;border-left-color:#D4D9E6;'
        f'margin-top:20px">{icon("info", size=18)}'
        f'<div><div style="font-weight:700;font-size:{T.font_xs}px">Not a diagnosis</div>'
        f'<div style="font-size:{T.font_xs}px;color:#5A6273">This is an educational screening aid. '
        'Always contact a qualified health professional.</div></div></div>',
        unsafe_allow_html=True,
    )
