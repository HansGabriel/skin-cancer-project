"""The nav keys — plain words, no emoji.

Emoji render from a different font at a different weight on every platform, and
on the Pi they fall back to a flat monochrome glyph. Words are legible for
everyone, translate cleanly, and keep the whole bar in one typeface.

The keys now sit at the foot of the dark instrument band rather than under the
page. Styling for that lives in ``theme.css`` under ``.st-key-epv-nav``; this
module owns *which* keys exist and what they do.

**There is deliberately no Exit key here.** It lived here briefly and cost more
than it bought. Six keys across the 45% band leaves ~77px each, which is narrower
than the word "New check" renders — that is the ``New checkSaved`` collision. It
also carried the only ``help=`` on any button in the app, which in Streamlit 1.57
wraps the ``<button>`` in two extra divs and broke every ``.stButton > button``
rule in the stylesheet, so the key painted near-white text on a white background
and lost its 74px height. Exit now lives only in Settings, behind the staff
passcode, which is where staff look for it anyway.
"""

from __future__ import annotations

import streamlit as st

from backend.assistant import kb_is_live
from navigation import Route, current_route, navigate

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
    cols = st.columns(len(nav_items), gap="small")
    for col, (label, route, here) in zip(cols, nav_items):
        if col.button(
            label,
            key=f"nav_{route}",
            use_container_width=True,
            type="primary" if active in here else "secondary",
        ):
            if route == "assistant":
                # The key means "ask about what I have now". Buttons on a
                # specific scan pin that scan; the key must unpin, or it
                # silently answers for a saved scan opened minutes ago.
                from views.assistant_view import clear_scan_context

                clear_scan_context()
            navigate(route)


# Kept as the old name so nothing that still imports it breaks mid-refactor.
render_bottom_nav = render_nav
