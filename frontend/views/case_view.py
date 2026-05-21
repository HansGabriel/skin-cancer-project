from __future__ import annotations

import json

import streamlit as st

from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link
from components.urgency_pill import render_urgency_from_band
from navigation import navigate
from services.storage import get_storage


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
        st.markdown(f"### {case.name}")
        if case.body_site:
            st.caption(case.body_site)
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
                render_urgency_from_band(s.risk_band)
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
