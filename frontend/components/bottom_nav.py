"""Persistent bottom navigation bar (all routes, PC and kiosk)."""

from __future__ import annotations

import streamlit as st

from backend.assistant import kb_is_live
from components.mobile_frame import mobile_frame
from navigation import Route, current_route, navigate


def _items() -> tuple[tuple[str, Route], ...]:
    items: list[tuple[str, Route]] = [
        ("🏠\nHome", "home"),
        ("📷\nScan", "camera"),
        ("📁\nHistory", "history"),
    ]
    # Assist appears only when doctor-reviewed answers exist — never a dead tab.
    if kb_is_live():
        items.append(("💬\nAssist", "assistant"))
    items.append(("⚙\nSettings", "settings"))
    return tuple(items)


def render_bottom_nav() -> None:
    active = current_route()
    nav_items = _items()
    with mobile_frame():
        st.divider()
        cols = st.columns(len(nav_items))
        for col, (label, route) in zip(cols, nav_items):
            is_active = active == route
            if col.button(
                label,
                key=f"nav_{route}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ) and not is_active:
                navigate(route)
