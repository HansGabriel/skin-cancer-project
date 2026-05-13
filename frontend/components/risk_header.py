"""Composite risk headline."""

from __future__ import annotations

import streamlit as st


def render_risk_header(composite: float, band: str) -> None:
    color = {"low": "green", "moderate": "orange", "high": "red"}.get(band, "blue")
    label = band.upper()
    st.markdown(
        f"### Screening composite: :{color}[**{composite:.0f}** / 100] — _{label} concern band_"
    )
    st.caption("Composite mixes CNN malignant probability with ABCDE tiers (A–D). Not a clinical score.")
