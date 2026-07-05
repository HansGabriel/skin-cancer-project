from __future__ import annotations

from typing import Any

import streamlit as st

from theme.tokens import TOKENS as T

_LABELS = {"A": "A-Asymmetry", "B": "B-Border", "C": "C-Colour", "D": "D-Diameter", "E": "E-Evolving"}
_TIER = {0: ("NORMAL ↓", T.success, "#111"), 1: ("BORDERLINE", T.warning, "#111"), 2: ("SUSPICIOUS ↑", T.urgent, "#fff")}


def render_abcde_row(abcde: dict[str, Any] | None) -> None:
    if not abcde:
        st.markdown(
            '<div class="ds-card"><p class="ds-section-title">ABCDE · educational</p>'
            f'<p style="margin:0;color:{T.text_muted}">Skipped — no stable lesion mask.</p></div>',
            unsafe_allow_html=True,
        )
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
        cap = f'<div style="font-size:{T.font_2xs}px;color:{T.text_muted};margin-top:4px">{detail}</div>' if detail else ""
        chips.append(
            f'<div style="flex:1;min-width:64px;background:#fff;border:1px solid {T.outline};'
            f'border-radius:12px;padding:10px;text-align:center;box-shadow:{T.shadow_sm}">'
            f'<div style="font-size:{T.chip_font}px;color:{T.text_muted}">{_LABELS[letter]}</div>'
            f'<div style="font-size:{T.stat_font}px;font-weight:700">{vstr}</div>'
            f'<span style="font-size:{T.font_2xs}px;padding:2px 6px;border-radius:999px;background:{bg};color:{fg}">'
            f'<span class="ds-tier-dot" style="background:{fg};opacity:.7"></span>{pill}</span>'
            f"{cap}</div>"
        )
    st.markdown(
        '<div class="ds-card"><p class="ds-section-title">ABCDE · educational</p>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px">{"".join(chips)}</div></div>',
        unsafe_allow_html=True,
    )
