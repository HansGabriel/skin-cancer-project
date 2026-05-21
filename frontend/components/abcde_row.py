from __future__ import annotations

from typing import Any

import streamlit as st

from theme.tokens import TOKENS as T

_LABELS = {"A": "A-Asymmetry", "B": "B-Border", "C": "C-Colour", "D": "D-Diameter", "E": "E-Evolving"}
_TIER = {0: ("NORMAL ↓", T.success, "#111"), 1: ("BORDERLINE", T.warning, "#111"), 2: ("SUSPICIOUS ↑", T.urgent, "#fff")}


def render_abcde_row(abcde: dict[str, Any] | None) -> None:
    st.markdown(f'<p style="font-weight:600">ABCDE Adapted/Educational</p>', unsafe_allow_html=True)
    if not abcde:
        st.info("ABCDE skipped — no stable lesion mask.")
        return
    chips = []
    for letter in "ABCDE":
        d = abcde.get(letter, {})
        val = d.get("value")
        vstr = "—" if val is None else (f"{val:.2f}" if isinstance(val, float) else str(val))
        if d.get("verdict") == "needs history":
            pill, bg, fg = "NEEDS HISTORY", T.info, "#fff"
        else:
            pill, bg, fg = _TIER.get(int(d.get("tier", 0)), ("—", T.outline, T.text))
        detail = d.get("detail", "")
        cap = f'<div style="font-size:9px;color:{T.text_muted};margin-top:4px">{detail}</div>' if detail else ""
        chips.append(
            f'<div style="flex:1;min-width:64px;background:{T.surface};border:1px solid {T.outline};'
            f'border-radius:12px;padding:10px;text-align:center">'
            f'<div style="font-size:10px;color:{T.text_muted}">{_LABELS[letter]}</div>'
            f'<div style="font-size:18px;font-weight:700">{vstr}</div>'
            f'<span style="font-size:9px;padding:2px 6px;border-radius:999px;background:{bg};color:{fg}">{pill}</span>'
            f"{cap}</div>"
        )
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:8px">{"".join(chips)}</div>', unsafe_allow_html=True)
