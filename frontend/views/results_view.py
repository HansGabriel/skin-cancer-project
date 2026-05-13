"""Latest pipeline output (risk, ABCDE, CNN, Grad-CAM, 7-class)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.contracts import ScanResult
from components.abcde_card import render_abcde_grid
from components.gradcam_viewer import render_gradcam_viewer
from components.prob_bars import render_seven_class_expander, render_three_class_probs
from components.risk_header import render_risk_header


def _render_scan_result_card(sr: ScanResult) -> None:
    urgency_color = {"LOW CONCERN": "green", "IMPORTANT": "orange", "URGENT": "red"}
    color = urgency_color.get(sr.urgency, "blue")
    st.markdown(f"### :{color}[{sr.icon} {sr.urgency}]")
    st.metric(
        "Detected",
        sr.label.upper().replace("_", " "),
        f"{sr.confidence:.1f}% confidence",
    )


def render_results_tab(backend) -> None:
    st.subheader("Latest result")
    pl = st.session_state.get("last_result")
    if not pl:
        st.info("Run a scan from the **Scan** tab to populate results.")
        return

    if pl.get("error") and not pl.get("scan_result"):
        st.error(pl["error"])
        return

    if pl.get("blocked") and not pl.get("scan_result"):
        st.error("Image did not pass quality checks.")
        for r in pl.get("quality", {}).get("reasons", []):
            st.warning(r)
        st.caption(
            "Tip: open **Settings**, turn off **Block run when quality checks fail**, then run again "
            "— the model will still run and these checks will show as warnings on Results."
        )
        return

    if "composite" in pl and "risk_band" in pl:
        render_risk_header(float(pl["composite"]), str(pl["risk_band"]))

    if pl.get("quality_warnings"):
        for w in pl["quality_warnings"]:
            st.warning(w)

    sr = pl.get("scan_result")
    if isinstance(sr, ScanResult):
        _render_scan_result_card(sr)

    left, right = st.columns([1.1, 1.0])
    with left:
        rgb = pl.get("rgb")
        render_gradcam_viewer(
            rgb,
            pl.get("gradcam_overlay_jpg"),
            pl.get("gradcam_disclaimer"),
        )
    with right:
        render_abcde_grid(pl.get("abcde"))

    if isinstance(sr, ScanResult):
        render_three_class_probs(sr.probs)
        keras_path = str(st.session_state.get("SKIN_KERAS_PATH_UI", ""))
        render_seven_class_expander(pl.get("seven_class_probs"), keras_path or "(not set)")

        st.info(f"Recommendation: {sr.action}")
        st.warning("For screening only — not a medical diagnosis.")

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("Save case (stub)", key="save_case_btn"):
            st.toast("Saved (stub — Tier 2 persistence not wired yet).")
    with c2:
        if st.button("Clear result", key="clear_result_btn"):
            st.session_state.pop("last_result", None)
            st.rerun()

    st.divider()
    st.subheader("Scan history")
    if st.button("Refresh history", key="log_refresh_results"):
        rows, warn = backend.fetch_log()
        if warn:
            st.warning(warn)
        if rows:
            try:
                st.dataframe(pd.json_normalize(rows), width="stretch")
            except Exception:  # noqa: BLE001
                st.dataframe(rows, width="stretch")
        elif not warn:
            st.info("No rows logged yet.")
