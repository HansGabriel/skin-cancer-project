from __future__ import annotations

from typing import Any

import streamlit as st

from theme.tokens import TOKENS as T

_LABELS = {"A": "A-Asymmetry", "B": "B-Border", "C": "C-Colour", "D": "D-Diameter", "E": "E-Evolving"}
# EPIVUE pill tones: soft tint background + strong text (matches the device concept).
_TIER = {
    0: ("NORMAL", T.success_tint, T.success),
    1: ("BORDERLINE", T.warning_tint, T.warning),
    2: ("SUSPICIOUS", T.urgent_tint, T.urgent),
}


def render_abcde_row(abcde: dict[str, Any] | None, *, show_values: bool = False) -> None:
    """ABCDE warning-sign chips.

    Default (user-facing) view shows letter + verdict pill only — the raw scores
    (0.08 / 1.50 / 15.13 …) mean nothing to ordinary users. Pass
    ``show_values=True`` to include them (used in the Technical details expander).
    """
    if not abcde:
        st.markdown(
            '<div class="ds-card"><p class="ds-section-title">ABCDE warning signs</p>'
            f'<p style="margin:0;color:{T.text_muted}">Couldn\'t outline the spot clearly — ABCDE check skipped.</p></div>',
            unsafe_allow_html=True,
        )
        return
    chips = []
    for letter in "ABCDE":
        d = abcde.get(letter, {})
        val = d.get("value")
        vstr = "—" if val is None else (f"{val:.2f}" if isinstance(val, float) else str(val))
        if d.get("verdict") == "needs history":
            # First scan of a case: Evolving cannot be assessed — say so, never "normal".
            pill, bg, fg = "NOT ASSESSED — FIRST VISIT", T.info_tint, T.info
        else:
            pill, bg, fg = _TIER.get(int(d.get("tier", 0)), ("—", T.outline, T.text))
        detail = d.get("detail", "")
        cap = (
            f'<div style="font-size:{T.font_2xs}px;color:{T.text_muted};margin-top:4px">{detail}</div>'
            if detail and show_values
            else ""
        )
        value_row = (
            f'<div style="font-size:{T.stat_font}px;font-weight:700">{vstr}</div>' if show_values else ""
        )
        chips.append(
            f'<div style="flex:1;min-width:64px;background:#fff;border:1px solid {T.outline};'
            f'border-radius:12px;padding:10px;text-align:center;box-shadow:{T.shadow_sm}">'
            f'<div style="font-size:{T.chip_font}px;color:{T.text_muted}">{_LABELS[letter]}</div>'
            f"{value_row}"
            f'<span style="font-size:{T.font_2xs}px;padding:2px 6px;border-radius:999px;background:{bg};color:{fg}">'
            f'<span class="ds-tier-dot" style="background:{fg};opacity:.7"></span>{pill}</span>'
            f"{cap}</div>"
        )
    caption = (
        f'<p style="margin:0 0 8px;color:{T.text_muted};font-size:{T.font_xs}px">'
        "A quick visual checklist doctors use for moles.</p>"
        if not show_values
        else ""
    )
    st.markdown(
        '<div class="ds-card"><p class="ds-section-title">ABCDE warning signs</p>'
        f'{caption}<div style="display:flex;flex-wrap:wrap;gap:8px">{"".join(chips)}</div></div>',
        unsafe_allow_html=True,
    )
