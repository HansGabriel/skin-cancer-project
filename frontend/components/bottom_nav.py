"""Persistent bottom navigation bar (all routes, PC and kiosk)."""

from __future__ import annotations

import streamlit as st

from components.mobile_frame import mobile_frame
from navigation import Route, current_route, navigate

_ITEMS: tuple[tuple[str, Route], ...] = (
    ("🏠\nHome", "home"),
    ("📷\nScan", "camera"),
    ("📁\nHistory", "history"),
    ("💬\nAssist", "assistant"),
    ("⚙\nSettings", "settings"),
)


def render_bottom_nav() -> None:
    active = current_route()
    with mobile_frame():
        st.divider()
        cols = st.columns(len(_ITEMS))
        for col, (label, route) in zip(cols, _ITEMS):
            is_active = active == route
            if col.button(
                label,
                key=f"nav_{route}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ) and not is_active:
                navigate(route)
