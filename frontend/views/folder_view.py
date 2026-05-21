from __future__ import annotations

import streamlit as st

from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link
from navigation import navigate
from services.storage import get_storage


@st.dialog("New case")
def _new_case(folder_id: str) -> None:
    name = st.text_input("Case name", "New lesion")
    site = st.selectbox("Body site", ["arm", "leg", "trunk", "face", "scalp", "hand", "foot", "other"])
    if st.button("Create") and name.strip():
        c = get_storage().create_case(folder_id, name.strip(), body_site=site)
        navigate("case", selected_case_id=c.id, selected_folder_id=folder_id)


def render_folder_view() -> None:
    fid = st.session_state.get("selected_folder_id")
    if not fid:
        navigate("history")
        return
    store = get_storage()
    folder = store.get_folder(fid)
    if not folder:
        navigate("history")
        return
    with mobile_frame():
        if render_back_link("History", key="fold_back"):
            navigate("history")
        st.markdown(f"### {folder.name}")
        if st.button("+ New case"):
            _new_case(fid)
        for c in store.list_cases(fid):
            n = len(store.list_scans(c.id))
            if st.button(f"{c.name} ({n} scans)", key=f"fc_{c.id}", use_container_width=True):
                navigate("case", selected_case_id=c.id, selected_folder_id=fid)
