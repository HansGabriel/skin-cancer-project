"""Saved spots — one row per case, newest verdict on the right.

A staff route (``services.auth.STAFF_ROUTES``): these are photographs of
people's skin, and ``docs/PRIVACY.md`` is the policy this enforces.

The old two-column folder tree needed a mental model of folders-then-cases
before anything was visible. This is one flat list of spots, with the folders
reduced to counts along the top, because "which spot" is the question anyone
opening this screen is actually asking.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st

from components.actions import actions_slot
from components.instrument import render_head
from navigation import navigate
from services.format import tone_colors
from services.kiosk import is_kiosk
from services.storage import get_storage
from services.verdict import verdict_for_saved_scan


@st.dialog("New folder")
def _new_folder() -> None:
    name = st.text_input("Name", "My scans")
    if st.button("Create") and name.strip():
        f = get_storage().create_folder(name.strip())
        navigate("folder", selected_folder_id=f.id)


def _thumb(path: Path) -> str:
    """Data URI for a saved scan's image, or empty when it cannot be read."""
    try:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return ""


def _render_row(case, scans, store, *, key: str) -> bool:
    latest = scans[-1]
    v = verdict_for_saved_scan(latest)
    ink, fill = tone_colors(v.tone)
    uri = _thumb(store.root / latest.image_path)
    bg = f"background-image:url('{uri}')" if uri else ""
    when = f"{len(scans)} scan{'s' if len(scans) != 1 else ''} · last {latest.taken_at[:10]}"
    row, action = st.columns([8, 2], gap="small")
    with row:
        st.markdown(
            f'<div class="ds-row"><span class="ds-thumb" style="{bg};'
            f'box-shadow:0 0 0 2px {ink}"></span>'
            f'<span class="ds-row-grow"><span style="display:block;font-weight:600">'
            f"{html.escape(case.name)}</span>"
            f'<span style="display:block;color:#5B6675;margin-top:2px">{when}</span></span>'
            f'<span class="ds-pill" style="background:{fill};color:{ink}">{v.headline}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with action:
        return action.button("Open", key=key, use_container_width=True)


def render_history_view() -> None:
    store = get_storage()
    folders = store.list_folders()
    render_head("Saved spots · this device only", "Saved spots")

    if not folders:
        st.markdown(
            '<div class="ds-mid"><p class="ds-lede">Nothing saved yet. After a check, '
            'choose "Save this scan" to keep a photo to compare against later.</p></div>',
            unsafe_allow_html=True,
        )
        with actions_slot():
            if st.button(
                "Add a new check", type="primary", key="hist_new", use_container_width=True
            ):
                navigate("camera")
        return

    # Folder counts as chips. Display only — the list below is already flat,
    # and a filter nobody can see the effect of is worse than no filter.
    rows: list[tuple] = []
    for f in folders:
        for c in store.list_cases(f.id):
            scans = store.list_scans(c.id)
            if scans:
                rows.append((f, c, scans))
    chips = [f'<span class="ds-chip is-on">All spots · {len(rows)}</span>']
    for f in folders:
        n = sum(1 for rf, _c, _s in rows if rf.id == f.id)
        chips.append(f'<span class="ds-chip">{html.escape(f.name)} · {n}</span>')
    st.markdown(f'<div class="ds-chiprow">{"".join(chips)}</div>', unsafe_allow_html=True)

    # No physical keyboard on the kiosk — hide the search box there.
    q = ""
    if not is_kiosk():
        st.text_input(
            "Search saved spots",
            placeholder="Search saved spots",
            key="hist_search",
            label_visibility="collapsed",
        )
        q = (st.session_state.get("hist_search") or "").strip().lower()

    shown = 0
    for folder, case, scans in rows:
        if q and q not in case.name.lower() and q not in folder.name.lower():
            continue
        shown += 1
        if _render_row(case, scans, store, key=f"hist_{case.id}"):
            navigate("case", selected_case_id=case.id, selected_folder_id=folder.id)
    if not shown:
        st.markdown('<p class="ds-reason">No spots match that search.</p>', unsafe_allow_html=True)

    with actions_slot():
        add, folder_btn = st.columns([3, 2], gap="small")
        with add:
            if st.button(
                "Add a new check", type="primary", key="hist_add", use_container_width=True
            ):
                navigate("camera")
        with folder_btn:
            if st.button("New folder", key="hist_newfolder", use_container_width=True):
                _new_folder()
