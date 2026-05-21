from __future__ import annotations

import streamlit as st


def render_primary_button(label: str, *, key: str) -> bool:
    return st.button(label, type="primary", use_container_width=True, key=key)


def render_back_link(label: str, *, key: str) -> bool:
    return st.button(f"← {label}", key=key)
