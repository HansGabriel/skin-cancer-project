"""Home: what this does, and one button to start.

The screen deliberately holds nothing else. It used to list saved folders,
recent scans and their thumbnails — which was both a crowded first screen and a
privacy hole: ``history`` is a staff route behind the passcode, but the same
photos were being previewed here with no gate at all. The saved scans are one
nav key away for anyone who is allowed to see them.
"""

from __future__ import annotations

import streamlit as st

from components.actions import actions_slot
from components.instrument import render_head
from navigation import navigate
from services.auth import staff_area_reachable

# One line each. On a 600px-tall kiosk panel these compete with the button that
# actually starts the check, so they say the minimum and stop.
_STEPS = (
    "Fill the ring with the spot and hold still",
    "Tap once to take the photo",
    "Read what to do next, in plain words",
)


def _render_steps() -> None:
    rows = "".join(
        f'<div class="ds-row"><span class="ds-row-num">{i}</span>'
        f'<span class="ds-row-grow">{text}</span></div>'
        for i, text in enumerate(_STEPS, 1)
    )
    st.markdown(f'<div class="ds-mid">{rows}</div>', unsafe_allow_html=True)


def render_home_view() -> None:
    render_head(
        "Skin check · takes about a minute",
        "Check a spot on your skin",
        "Nothing you photograph leaves this device. No name, no account.",
    )
    _render_steps()

    with actions_slot():
        show_saved = staff_area_reachable()
        if show_saved:
            start, saved = st.columns([3, 2], gap="small")
        else:
            start, saved = st.container(), None
        with start:
            if st.button(
                "Start a skin check", type="primary", key="home_scan", use_container_width=True
            ):
                st.session_state.pop("capture_image_bytes", None)
                navigate("camera")
        if saved is not None:
            with saved:
                if st.button("Saved spots", key="home_saved", use_container_width=True):
                    navigate("history")
