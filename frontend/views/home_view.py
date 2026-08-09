"""Home: what this does, and one button to start.

The old version stacked three empty things — a blank grey square, an empty
search box, and a "No scans yet" card — so a fresh install looked broken. Now
the aperture sits at rest with an instruction in it, and the history section
only appears once there is history.
"""

from __future__ import annotations

import streamlit as st

from components.aperture import render_aperture
from components.app_bar import render_app_bar, render_disclaimer_footer
from components.folder_card import render_folder_card
from components.mobile_frame import mobile_frame
from components.primary_button import render_primary_button
from components.scan_row import render_scan_row
from navigation import navigate
from services.kiosk import is_kiosk
from services.storage import get_storage
from theme.tokens import TOKENS as T

# One line each. On a 600px-tall kiosk panel these compete with the button that
# actually starts the check, so they say the minimum and stop.
_STEPS = (
    "Fill the ring with the spot and hold still",
    "Tap once to take the photo",
    "Read what to do next, in plain words",
)


def _render_steps() -> None:
    rows = "".join(
        f'<div style="display:flex;gap:{T.space_12}px;align-items:center;'
        f'padding:{T.space_4}px 0;font-size:{T.font_sm}px">'
        f'<span style="flex:0 0 20px;height:20px;border-radius:999px;background:{T.surface};'
        f'color:{T.text_muted};font-size:{T.font_2xs}px;font-weight:700;display:flex;'
        f'align-items:center;justify-content:center">{i}</span>'
        f"<span>{text}</span></div>"
        for i, text in enumerate(_STEPS, 1)
    )
    st.markdown(rows, unsafe_allow_html=True)


def render_home_view() -> None:
    with mobile_frame():
        render_app_bar()
        left, right = st.columns([5, 6], gap="large")
        with left:
            render_aperture(hint="Put the spot inside the ring")
        with right:
            st.markdown(
                f'<p style="font-size:{T.font_lg}px;font-weight:700;letter-spacing:-.02em;'
                f'margin:0 0 {T.space_4}px">Check a spot on your skin</p>'
                f'<p style="color:{T.text_muted};font-size:{T.font_sm}px;margin:0 0 {T.space_12}px">'
                "Takes about a minute. Nothing leaves this device.</p>",
                unsafe_allow_html=True,
            )
            _render_steps()
            st.markdown('<div class="ds-gap"></div>', unsafe_allow_html=True)
            if render_primary_button("Start a skin check", key="home_scan"):
                navigate("camera")

        store = get_storage()
        folders = store.list_folders()
        recent = store.latest_scans_global(4)
        if folders or recent:
            st.markdown('<hr class="ds-rule">', unsafe_allow_html=True)
            st.markdown('<p class="ds-section-title">Saved spots</p>', unsafe_allow_html=True)
            # No physical keyboard on the kiosk — hide the search box there.
            q = ""
            if not is_kiosk():
                st.text_input(
                    "Search saved spots",
                    placeholder="Search saved spots",
                    key="home_search",
                    label_visibility="collapsed",
                )
                q = (st.session_state.get("home_search") or "").strip().lower()
            filtered = [f for f in folders if not q or q in f.name.lower()]
            if filtered:
                cols = st.columns(min(4, len(filtered)))
                for col, folder in zip(cols, filtered[:4]):
                    with col:
                        if render_folder_card(folder, key=f"home_f_{folder.id}"):
                            navigate("folder", selected_folder_id=folder.id)
            for scan, case, folder in recent:
                if q and q not in case.name.lower() and q not in folder.name.lower():
                    continue
                if render_scan_row(scan, case, folder, key=f"home_s_{scan.id}"):
                    navigate("case", selected_case_id=case.id, selected_folder_id=folder.id)
            if st.button("See all saved spots", key="home_all_hist"):
                navigate("history")

        render_disclaimer_footer()
