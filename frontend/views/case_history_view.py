"""Legacy tab stub — use History screen (route=history)."""

from __future__ import annotations

import streamlit as st

from navigation import navigate


def render_case_history_tab() -> None:
    st.info("Cases moved to the **History** screen in DermaScan v2.")
    if st.button("Open History"):
        navigate("history")
