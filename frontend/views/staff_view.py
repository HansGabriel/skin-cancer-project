"""The clinical readout — raw numbers, for whoever is supervising the device.

This was an ``st.expander("Details for staff")`` on the results screen, open to
anyone who tapped it, while ``history`` / ``case`` / ``settings`` next door were
behind the passcode. It is a route now, which means ``services.auth`` gates it
like the rest (see ``STAFF_ROUTES``) and it fills the page band the way the
design's overlay does, with the instrument still visible beside it.

Everything here is deliberately in the vocabulary the visitor screens avoid:
asymmetry/border/colour/diameter, calibrated confidences, model file, timings.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from backend.contracts import ScanResult
from components.actions import actions_slot
from components.image_compare import render_image_compare
from components.instrument import render_head
from navigation import navigate
from services.eigencam import cam_model_path
# NOT services.format.fmt_pct: it treats a value <= 1.0 as a fraction and
# multiplies by 100, so a genuine 0.6% malignant probability printed as
# "60.0%" on the clinical readout. ScanResult.probs is already 0-100.
from services.format import field_ink
from services.pipeline import trust_line
from services.scan_flow import build_attention_overlay
from theme.tokens import TOKENS as T

_LETTER_NAMES = {
    "A": "Asymmetry",
    "B": "Border",
    "C": "Colour",
    "D": "Diameter mm",
    "E": "Evolving",
}
_PILL = {0: "NORMAL", 1: "BORDERLINE", 2: "SUSPICIOUS"}
_BAND_LABELS = {"benign": "Benign", "pre_cancerous": "Pre-cancerous", "malignant": "Malignant"}
_BAND_TIER = {"benign": 0, "pre_cancerous": 1, "malignant": 2}


def cam_available() -> bool:
    """Whether an attention view could be built at all on this device."""
    return cam_model_path().is_file()


def _render_abcde(abcde: dict | None) -> None:
    if not abcde:
        st.markdown(
            f'<p style="color:{T.field_muted}">The spot could not be outlined, so the '
            "five checks were skipped.</p>",
            unsafe_allow_html=True,
        )
        return
    rows = []
    for letter in "ABCDE":
        d = abcde.get(letter) or {}
        value = d.get("value")
        vstr = "—" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))
        if d.get("verdict") == "needs history":
            # First scan of a case: Evolving cannot be assessed — say so, never "normal".
            pill, ink = "NEEDS HISTORY", T.on_field_veil
        else:
            tier = int(d.get("tier", 0) or 0)
            pill, ink = _PILL.get(tier, "—"), field_ink(tier)
        rows.append(
            f'<div class="ds-staff-row"><span class="ds-staff-letter">{letter}</span>'
            f'<span class="ds-staff-name">{_LETTER_NAMES[letter]}</span>'
            f'<span class="ds-staff-value">{vstr}</span>'
            f'<span class="ds-staff-pill" style="color:{ink}">{pill}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_probs(probs: dict[str, float]) -> None:
    rows = []
    for key in ("benign", "pre_cancerous", "malignant"):
        pct = float(probs.get(key, 0.0))
        ink = field_ink(_BAND_TIER[key])
        rows.append(
            f'<div class="ds-staff-row" style="border-top:0">'
            f'<span class="ds-staff-name" style="max-width:140px">{_BAND_LABELS[key]}</span>'
            f'<span class="ds-staff-track"><span class="ds-staff-fill" '
            f'style="width:{max(1.5, pct):.1f}%;background:{ink}"></span></span>'
            f'<span class="ds-staff-value">{pct:.1f}%</span></div>'
        )
    st.markdown(
        f'<p class="ds-section-title" style="color:{T.field_muted};margin-top:{T.space_16}px">'
        "Model confidence (calibrated)</p>" + "".join(rows),
        unsafe_allow_html=True,
    )


def _render_meta(pl: dict, root: Path, model_path: str) -> None:
    bits = [line for line in (trust_line(pl),) if line]
    t_path = Path(model_path).resolve().parent / "temperature.json"
    if not t_path.is_file():
        t_path = root / "models" / "temperature.json"
    if t_path.is_file():
        try:
            t_val = float(json.loads(t_path.read_text()).get("T", 1.0))
        except (OSError, ValueError, TypeError):
            t_val = 1.0
        if t_val != 1.0:
            # Displayed confidence IS calibrated (backend/tflite_shared.apply_temperature);
            # the screening decision itself still uses raw probabilities.
            bits.append(f"temperature T = {t_val:.2f}")
    if bits:
        st.markdown(
            f'<div class="ds-staff-meta">{" · ".join(bits)}</div>', unsafe_allow_html=True
        )


def render_staff_view(*, root: Path, model_path: str) -> None:
    pl = st.session_state.get("last_result")
    sr = pl.get("scan_result") if isinstance(pl, dict) else None
    if not isinstance(sr, ScanResult):
        render_head("Clinical readout · staff only", "No scan to report on")
        with actions_slot():
            if st.button("Back", type="primary", key="staff_back_empty", use_container_width=True):
                navigate("results")
        return

    render_head("Clinical readout · staff only", "")
    _render_abcde(pl.get("abcde"))
    _render_probs(sr.probs)

    seven = pl.get("seven_class_probs")
    if seven:
        with st.expander("Detailed lesion-type breakdown"):
            from components.prob_bars import render_seven_class_bars

            render_seven_class_bars(seven)

    # The heatmap is built on request, not on every scan: it costs a second
    # full model pass plus a full-resolution blend.
    if pl.get("attention_overlay_jpg"):
        render_image_compare(pl.get("rgb"), pl.get("attention_overlay_jpg"), note=pl.get("attention_note"))
    elif not cam_available():
        st.caption("The attention view is not available on this device.")
    elif st.button("Show where the scanner looked", key="staff_attention", use_container_width=True):
        with st.spinner("Building the attention view…"):
            build_attention_overlay(pl, str(st.session_state.get("SKIN_KERAS_PATH_UI", "")))
        st.session_state["last_result"] = pl
        st.rerun()

    if pl.get("rgb_before") is not None and pl.get("rgb_analysis") is not None:
        with st.expander("Preprocessing (before / after — ABCDE path only)"):
            c1, c2 = st.columns(2)
            c1.image(pl["rgb_before"], caption="Original (classifier input)", width="stretch")
            c2.image(pl["rgb_analysis"], caption="Enhanced (ABCDE/segmentation)", width="stretch")

    _render_meta(pl, root, model_path)

    with actions_slot():
        if st.button("Close", type="primary", key="staff_close", use_container_width=True):
            navigate("results")
