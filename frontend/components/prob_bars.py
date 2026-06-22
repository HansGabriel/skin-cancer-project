"""3-class and 7-class horizontal probability bars."""

from __future__ import annotations

import streamlit as st

from services.format import fmt_pct
from services.gradcam import HAM7_LABELS
from theme.tokens import TOKENS as T

_PRETTY7 = {
    "akiec": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
_LABELS3 = {"benign": "Benign", "pre_cancerous": "Pre-Cancerous", "malignant": "Malignant"}


def _bar(label: str, pct: float, *, fill: str | None = None) -> str:
    w = max(0.0, min(100.0, pct))
    style = f'style="width:{w}%;background:{fill}"' if fill else f'style="width:{w}%"'
    return (
        f'<div style="display:flex;align-items:center;gap:12px;margin:8px 0">'
        f'<span style="min-width:110px;font-size:13px">{label}</span>'
        f'<div class="ds-prob-track"><div class="ds-prob-fill" {style}></div></div>'
        f'<span style="min-width:48px;text-align:right">{fmt_pct(pct)}</span></div>'
    )


def render_three_class_probs(probs: dict[str, float]) -> None:
    # Malignant gets the urgent hue so the eye lands on the important class.
    fills = {"malignant": f"linear-gradient(90deg,{T.urgent},{T.urgent})"}
    bars = "".join(
        _bar(_LABELS3[k], float(probs.get(k, 0)), fill=fills.get(k))
        for k in ("benign", "pre_cancerous", "malignant")
    )
    st.markdown(
        f'<div class="ds-card"><p class="ds-section-title">CNN · 3-class</p>{bars}</div>',
        unsafe_allow_html=True,
    )


def render_seven_class_expander(seven: dict[str, float] | None, keras_path: str) -> None:
    with st.expander("Show differential (advanced)"):
        if not seven:
            st.info(f"No 7-class head. Set SKIN_KERAS_PATH (currently: {keras_path}).")
            return
        st.markdown(
            "".join(_bar(_PRETTY7.get(dx, dx), float(seven.get(dx, 0))) for dx in HAM7_LABELS),
            unsafe_allow_html=True,
        )
