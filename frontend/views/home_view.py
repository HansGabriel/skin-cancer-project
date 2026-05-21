from __future__ import annotations

import streamlit as st

from components.app_bar import render_app_bar, render_disclaimer_footer
from components.empty_state import render_empty_state
from components.folder_card import render_folder_card
from components.mobile_frame import mobile_frame
from components.primary_button import render_primary_button
from components.scan_row import render_scan_row
from components.viewfinder import render_viewfinder_placeholder
from navigation import navigate
from services.storage import get_storage


def render_home_view() -> None:
    with mobile_frame():
        render_app_bar()
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        render_viewfinder_placeholder()
        if render_primary_button("SCAN LESION", key="home_scan"):
            navigate("camera")
        render_disclaimer_footer()
        st.markdown("### History")
        st.text_input("Search", placeholder="🔍", key="home_search", label_visibility="collapsed")
        store = get_storage()
        folders = store.list_folders()
        q = (st.session_state.get("home_search") or "").strip().lower()
        if not folders and not store.latest_scans_global(1):
            render_empty_state("No scans yet", "Your scans will appear here.")
            return
        filtered = [f for f in folders if not q or q in f.name.lower()]
        if filtered:
            st.markdown('<div class="ds-history-scroll">', unsafe_allow_html=True)
            cols = st.columns(min(4, len(filtered)))
            for col, folder in zip(cols, filtered[:4]):
                with col:
                    if render_folder_card(folder, key=f"home_f_{folder.id}"):
                        navigate("folder", selected_folder_id=folder.id)
            st.markdown("</div>", unsafe_allow_html=True)
        for scan, case, folder in store.latest_scans_global(4):
            if q and q not in case.name.lower() and q not in folder.name.lower():
                continue
            if render_scan_row(scan, case, folder, key=f"home_s_{scan.id}"):
                navigate("case", selected_case_id=case.id, selected_folder_id=folder.id)
        if st.button("View all history", key="home_all_hist"):
            navigate("history")
