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
from services.kiosk import is_kiosk, request_quit

_CONFIRM_KEY = "_exit_armed"


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
    # The kiosk gets an Exit tab of its own. It used to live only behind the
    # staff passcode in Settings, which meant closing the kiosk needed either
    # the code or an SSH session — no good when someone just wants the screen
    # back. Two taps rather than one so a stray touch cannot end the demo.
    show_exit = is_kiosk()
    with mobile_frame():
        st.markdown('<hr class="ds-rule">', unsafe_allow_html=True)
        cols = st.columns(len(nav_items) + (1 if show_exit else 0))
        for col, (label, route) in zip(cols, nav_items):
            is_active = active == route
            if col.button(
                label,
                key=f"nav_{route}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.pop(_CONFIRM_KEY, None)
                if route == "assistant":
                    # The tab means "ask about what I have now". Buttons on a
                    # specific scan pin that scan; the tab must unpin, or it
                    # silently answers for a saved scan opened minutes ago.
                    from views.assistant_view import clear_scan_context

                    clear_scan_context()
                navigate(route)
        if show_exit:
            _render_exit(cols[-1])


def _render_exit(col) -> None:
    armed = st.session_state.get(_CONFIRM_KEY, False)
    label = "Tap again" if armed else "Exit"
    if col.button(
        label,
        key="nav_exit",
        use_container_width=True,
        type="primary" if armed else "secondary",
        help="Closes the kiosk and returns to the desktop.",
    ):
        if armed:
            st.session_state.pop(_CONFIRM_KEY, None)
            request_quit()
            st.stop()
        st.session_state[_CONFIRM_KEY] = True
        st.rerun()
