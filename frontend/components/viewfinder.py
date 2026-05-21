"""Viewfinder frame with corner brackets (offline SVG)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import streamlit as st

_BRACKET = '<svg width="24" height="24"><path d="M4 8V4H8" stroke="#B58CF0" stroke-width="2.5" fill="none"/></svg>'
_CORNERS = f"""<div style="position:absolute;inset:0;pointer-events:none;z-index:2;">
<div style="position:absolute;top:8px;left:8px">{_BRACKET}</div>
<div style="position:absolute;top:8px;right:8px;transform:rotate(90deg)">{_BRACKET}</div>
<div style="position:absolute;bottom:8px;right:8px;transform:rotate(180deg)">{_BRACKET}</div>
<div style="position:absolute;bottom:8px;left:8px;transform:rotate(270deg)">{_BRACKET}</div></div>"""
_CROSSHAIR = (
    '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
    'pointer-events:none;opacity:.25"><div style="width:40%;height:1px;background:#B58CF0"></div></div>'
)


def render_viewfinder_placeholder() -> None:
    st.markdown(
        f'<div class="ds-viewfinder">{_CROSSHAIR}<div class="ds-viewfinder-slot"></div>{_CORNERS}</div>',
        unsafe_allow_html=True,
    )


@contextmanager
def viewfinder_slot(*, shutter: bool = False) -> Generator[None, None, None]:
    """Wrap Streamlit widgets (camera/upload) inside the viewfinder frame."""
    cls = "ds-viewfinder ds-shutter-pulse" if shutter else "ds-viewfinder"
    st.markdown(f'<div class="{cls}"><div class="ds-viewfinder-slot">', unsafe_allow_html=True)
    yield
    st.markdown(f"</div>{_CORNERS}</div>", unsafe_allow_html=True)
