from __future__ import annotations

from datetime import datetime

import streamlit as st

from navigation import navigate


def format_app_bar_time(now: datetime | None = None) -> str:
    dt = now or datetime.now()
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{dt.minute:02d}{ampm} {dt.strftime('%a, %b')} {dt.day}, {dt.year}"


def render_app_bar(*, show_home: bool = False) -> None:
    cols = st.columns([1, 1, 1, 3] if show_home else [1, 1, 4])
    i = 0
    if show_home:
        if cols[i].button("🏠", key="app_bar_home"):
            navigate("home")
        i += 1
    if cols[i].button("⚙", key="app_bar_settings"):
        navigate("settings")
    if cols[i + 1].button("📁", key="app_bar_history"):
        navigate("history")
    st.markdown(f'<p class="ds-app-bar-time">{format_app_bar_time()}</p>', unsafe_allow_html=True)


def render_disclaimer_footer() -> None:
    st.markdown('<p class="ds-disclaimer">DISCLAIMER</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-disclaimer-sub">Contact a Health Professional</p>', unsafe_allow_html=True)
