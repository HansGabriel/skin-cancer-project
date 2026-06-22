"""Circular risk/confidence gauge — the results-screen hero (pure CSS, no JS)."""

from __future__ import annotations

import streamlit as st

from theme.tokens import TOKENS as T

# Risk band → (ring color, soft tint, urgency label).
_BAND = {
    "high": (T.urgent, T.urgent_tint, "URGENT"),
    "moderate": (T.warning, T.warning_tint, "IMPORTANT"),
    "low": (T.success, T.success_tint, "LOW CONCERN"),
}


def render_risk_ring(label: str, confidence: float, band: str) -> None:
    """Draw a conic-gradient ring filled to ``confidence`` %, colored by ``band``.

    The class label sits large in the center with the confidence beneath, and a
    small urgency chip under the ring.
    """
    color, tint, urgency = _BAND.get(band, (T.info, T.info_tint, "SCREENING"))
    pct = max(0.0, min(100.0, float(confidence)))
    display = label.upper().replace("_", " ")
    ring_bg = f"conic-gradient({color} {pct:.0f}%, {T.outline} {pct:.0f}%)"
    st.markdown(
        f'<div class="ds-ring-wrap">'
        f'<div class="ds-ring" style="background:{ring_bg}">'
        f'<div class="ds-ring-inner" style="background:{tint}">'
        f'<div class="ds-ring-label" style="color:{color}">{display}</div>'
        f'<div class="ds-ring-conf">{pct:.0f}% confidence</div>'
        f"</div></div>"
        f'<span style="margin-top:10px;padding:5px 14px;border-radius:999px;'
        f'background:{color};color:#fff;font-weight:700;font-size:{T.font_xs}px;'
        f'letter-spacing:.04em">{urgency}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
