"""Folder card for History / Home strips."""

from __future__ import annotations

import streamlit as st

from services.storage import Folder
from theme.tokens import TOKENS as T


def render_folder_card(folder: Folder, *, key: str) -> bool:
    """Render a folder card button; returns True when clicked."""
    color = folder.color or T.violet
    st.markdown(
        f'<div class="ds-folder-card" style="border-color:{color}55">'
        f'<span style="color:{color}">●</span> {folder.name}</div>',
        unsafe_allow_html=True,
    )
    return st.button(folder.name, key=key, use_container_width=True)
