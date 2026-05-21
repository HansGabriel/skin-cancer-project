"""Scan row for home history preview."""

from __future__ import annotations

import streamlit as st

from services.format import urgency_from_band
from services.storage import Case, Folder, Scan
from theme.tokens import TOKENS as T


def render_scan_row(scan: Scan, case: Case, folder: Folder, *, key: str) -> bool:
    label, color = urgency_from_band(scan.risk_band)
    st.markdown(
        f'<div class="ds-scan-row">'
        f'<span style="color:{T.text_muted}">{folder.name} ›</span> '
        f'<span>{case.name}</span> '
        f'<span style="font-size:10px;color:{color};font-weight:700">{label}</span> '
        f'<span style="color:{T.text_muted}">{scan.taken_at[:10]}</span></div>',
        unsafe_allow_html=True,
    )
    return st.button(f"View {case.name}", key=key, use_container_width=True)
