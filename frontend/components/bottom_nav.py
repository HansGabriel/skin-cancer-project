"""The nav keys — plain words, no emoji.

Emoji render from a different font at a different weight on every platform, and
on the Pi they fall back to a flat monochrome glyph. Words are legible for
everyone, translate cleanly, and keep the whole bar in one typeface.

The keys now sit at the foot of the dark instrument band rather than under the
page. Styling for that lives in ``theme.css`` under ``.st-key-epv-nav``; this
module owns *which* keys exist and what they do.
"""

from __future__ import annotations

import streamlit as st

from backend.assistant import kb_is_live
from navigation import Route, current_route, navigate
from services.kiosk import is_kiosk, request_quit

_CONFIRM_KEY = "_exit_armed"

# Routes that light the "New check" key: the whole capture flow is one
# destination as far as a visitor is concerned, even though it spans four
# screens and the result.
_CHECK_ROUTES: tuple[Route, ...] = ("camera", "reading", "results", "staff")
_SAVED_ROUTES: tuple[Route, ...] = ("history", "folder", "case")


def _items() -> tuple[tuple[str, Route, tuple[Route, ...]], ...]:
    """(label, destination, routes that count as "here")."""
    items: list[tuple[str, Route, tuple[Route, ...]]] = [
        ("Home", "home", ("home",)),
        ("New check", "camera", _CHECK_ROUTES),
        ("Saved", "history", _SAVED_ROUTES),
    ]
    # Assist appears only when doctor-reviewed answers exist — never a dead tab.
    if kb_is_live():
        items.append(("Questions", "assistant", ("assistant",)))
    items.append(("Settings", "settings", ("settings",)))
    return tuple(items)


def render_nav() -> None:
    active = current_route()
    nav_items = _items()
    # The kiosk gets an Exit key of its own. It used to live only behind the
    # staff passcode in Settings, which meant closing the kiosk needed either
    # the code or an SSH session — no good when someone just wants the screen
    # back. Two taps rather than one so a stray touch cannot end the demo.
    show_exit = is_kiosk()
    cols = st.columns(len(nav_items) + (1 if show_exit else 0), gap="small")
    for col, (label, route, here) in zip(cols, nav_items):
        if col.button(
            label,
            key=f"nav_{route}",
            use_container_width=True,
            type="primary" if active in here else "secondary",
        ):
            st.session_state.pop(_CONFIRM_KEY, None)
            if route == "assistant":
                # The key means "ask about what I have now". Buttons on a
                # specific scan pin that scan; the key must unpin, or it
                # silently answers for a saved scan opened minutes ago.
                from views.assistant_view import clear_scan_context

                clear_scan_context()
            navigate(route)
    if show_exit:
        _render_exit(cols[-1])


# Kept as the old name so nothing that still imports it breaks mid-refactor.
render_bottom_nav = render_nav


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
