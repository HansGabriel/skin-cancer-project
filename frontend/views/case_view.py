from __future__ import annotations

import json

import streamlit as st

from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link
from components.urgency_pill import render_verdict_pill
from navigation import navigate
from services.storage import get_storage
from services.verdict import verdict_for_saved_scan
from theme.tokens import TOKENS as T


def render_case_view() -> None:
    cid = st.session_state.get("selected_case_id")
    fid = st.session_state.get("selected_folder_id")
    if not cid:
        navigate("history")
        return
    store = get_storage()
    case = store.get_case(cid)
    if not case:
        navigate("history")
        return
    scans = store.list_scans(cid)
    with mobile_frame():
        if render_back_link("Back", key="case_back"):
            navigate("folder", selected_folder_id=fid) if fid else navigate("history")
        # EPIVUE-style case header: "CASE · BODY SITE" eyebrow above the case name.
        site = (case.body_site or "").upper()
        chip = f"CASE · {site}" if site else "CASE"
        st.markdown(
            f'<div style="font-size:{T.font_xs}px;letter-spacing:.12em;font-weight:700;'
            f'color:{T.text_muted};margin:4px 0 2px">{chip}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {case.name}")
        if not scans:
            st.info("No scans yet.")
            if st.button("+ Add scan"):
                st.session_state["pending_case_id"] = cid
                navigate("camera")
            return
        st.markdown("#### Timeline")
        cols = st.columns(min(4, len(scans)))
        for i, s in enumerate(scans):
            with cols[i % len(cols)]:
                st.image(str(store.root / s.image_path), use_container_width=True)
                st.caption(s.taken_at[:10])
                render_verdict_pill(verdict_for_saved_scan(s))
        if len(scans) >= 2:
            labels = [f"{s.taken_at[:10]}" for s in scans]
            d_vals, b_vals, a_vals = [], [], []
            for s in scans:
                ab = json.loads(s.abcd_json)
                for letter, out in (("D", d_vals), ("B", b_vals), ("A", a_vals)):
                    dv = ab.get(letter, {}).get("value")
                    out.append(float(dv) if isinstance(dv, (int, float)) else 0.0)
            st.markdown("#### Evolution timeline")
            st.line_chart(
                {"diameter_mm": d_vals, "border_score": b_vals, "asymmetry_score": a_vals},
                x=labels,
            )
            ia = st.selectbox("Scan A", range(len(scans)), format_func=lambda i: labels[i])
            ib = st.selectbox("Scan B", range(len(scans)), format_func=lambda i: labels[i])
            sa, sb = scans[ia], scans[ib]
            c1, c2 = st.columns(2)
            c1.image(str(store.root / sa.image_path))
            c2.image(str(store.root / sb.image_path))
            for letter in "ABCD":
                va = json.loads(sa.abcd_json).get(letter, {}).get("value")
                vb = json.loads(sb.abcd_json).get(letter, {}).get("value")
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    st.write(f"{letter} Δ: {vb - va:+.3f}")
            if sa.e_json:
                st.caption(f"E: {sa.e_json}")
        if st.button("+ Add scan to this case"):
            st.session_state["pending_case_id"] = cid
            navigate("camera")
        if st.button("Export CSV"):
            st.download_button("Download", store.export_case_csv(cid), file_name="case.csv")
