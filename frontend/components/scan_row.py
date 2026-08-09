"""Scan row for home history preview."""

from __future__ import annotations

import html

import streamlit as st

from services.format import tone_colors
from services.storage import Case, Folder, Scan
from services.verdict import verdict_for_saved_scan
from theme.tokens import TOKENS as T


def render_scan_row(scan: Scan, case: Case, folder: Folder, *, key: str) -> bool:
    v = verdict_for_saved_scan(scan)
    ink, _fill = tone_colors(v.tone)
    st.markdown(
        f'<div class="ds-scan-row" style="display:flex;align-items:center;gap:8px">'
        f'<span class="ds-tier-dot" style="background:{ink}"></span>'
        f'<span style="color:{T.text_muted}">{html.escape(folder.name)} ›</span> '
        f'<span style="font-weight:600">{html.escape(case.name)}</span> '
        f'<span style="font-size:{T.chip_font}px;color:{ink};font-weight:700">{v.headline}</span> '
        f'<span style="color:{T.text_muted};margin-left:auto">{scan.taken_at[:10]}</span></div>',
        unsafe_allow_html=True,
    )
    return st.button(f"View {html.escape(case.name)}", key=key, use_container_width=True)
