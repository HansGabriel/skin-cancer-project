"""Bottom navigation — plain words, no emoji.

Emoji render from a different font at a different weight on every platform, and
on the Pi they fall back to a flat monochrome glyph. Words are legible for
everyone, translate cleanly, and keep the whole bar in one typeface.
"""

from __future__ import annotations

import streamlit as st

from backend.assistant import kb_is_live
from components.mobile_frame import mobile_frame
from navigation import Route, current_route, navigate


def _items() -> tuple[tuple[str, Route], ...]:
    items: list[tuple[str, Route]] = [
        ("Home", "home"),
        ("New check", "camera"),
        ("Saved", "history"),
    ]
    # Assist appears only when doctor-reviewed answers exist — never a dead tab.
    if kb_is_live():
        items.append(("Questions", "assistant"))
    items.append(("Settings", "settings"))
    return tuple(items)


def render_bottom_nav() -> None:
    active = current_route()
    nav_items = _items()
    with mobile_frame():
        st.markdown('<hr class="ds-rule">', unsafe_allow_html=True)
        cols = st.columns(len(nav_items))
        for col, (label, route) in zip(cols, nav_items):
            is_active = active == route
            if col.button(
                label,
                key=f"nav_{route}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                navigate(route)
