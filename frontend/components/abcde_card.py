"""ABCDE letter metrics as Streamlit columns."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_abcde_grid(abcde: dict[str, Any] | None) -> None:
    st.subheader("ABCDE (adapted / educational)")
    if abcde is None:
        st.info("ABCDE skipped — could not isolate a stable lesion mask (try lighting and centering).")
        return
    cols = st.columns(5)
    order = ["A", "B", "C", "D", "E"]
    titles = {
        "A": "Asymmetry",
        "B": "Border",
        "C": "Colour",
        "D": "Diameter",
        "E": "Evolving",
    }
    for i, letter in enumerate(order):
        d = abcde[letter]
        with cols[i]:
            val = d["value"]
            vstr = "—" if val is None else str(val)
            st.metric(f"{letter} · {titles[letter]}", vstr, delta=d["verdict"][:14])
