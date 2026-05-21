from __future__ import annotations

import streamlit as st

from components.app_bar import render_app_bar
from components.case_row import render_case_row
from components.empty_state import render_empty_state
from components.folder_card import render_folder_card
from components.mobile_frame import mobile_frame
from navigation import navigate
from services.storage import get_storage


@st.dialog("New folder")
def _new_folder() -> None:
    name = st.text_input("Name", "My scans")
    if st.button("Create") and name.strip():
        f = get_storage().create_folder(name.strip())
        navigate("folder", selected_folder_id=f.id)


def render_history_view() -> None:
    with mobile_frame():
        render_app_bar()
        st.markdown("### History")
        st.text_input("Search", key="hist_search", placeholder="🔍", label_visibility="collapsed")
        if st.button("+ New folder"):
            _new_folder()
        store = get_storage()
        folders = store.list_folders()
        q = (st.session_state.get("hist_search") or "").strip().lower()
        if not folders:
            render_empty_state("No history", "Save a scan from Results to get started.")
            return
        left, right = st.columns([1, 2])
        with left:
            for f in folders:
                if q and q not in f.name.lower():
                    continue
                if render_folder_card(f, key=f"hf_{f.id}"):
                    navigate("folder", selected_folder_id=f.id)
        with right:
            for f in folders:
                if q and q not in f.name.lower():
                    continue
                st.markdown(f"**{f.name}**")
                for c in store.list_cases(f.id):
                    if q and q not in c.name.lower() and (not c.body_site or q not in c.body_site.lower()):
                        continue
                    scans = store.list_scans(c.id)
                    urg = scans[-1].risk_band if scans else "low"
                    dt = scans[-1].taken_at[:10] if scans else "—"
                    if render_case_row(c, urgency=urg, date=dt, key=f"hc_{c.id}"):
                        navigate("case", selected_case_id=c.id, selected_folder_id=f.id)
