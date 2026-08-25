from __future__ import annotations

import json

import streamlit as st

from backend.assistant import kb_is_live
from components.actions import actions_slot
from components.instrument import render_head
from components.urgency_pill import render_verdict_pill
from navigation import navigate
from services.storage import get_storage
from services.verdict import verdict_for_saved_scan
from views.assistant_view import ask_about_saved_scan


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
    site = (case.body_site or "").lower()
    render_head(f"Saved spot · {site}" if site else "Saved spot", case.name)
    if not scans:
        st.info("No scans yet.")
        with actions_slot():
            if st.button("Add a check", type="primary", key="case_add_first", use_container_width=True):
                st.session_state["pending_case_id"] = cid
                navigate("camera")
        return
    st.markdown('<p class="ds-section-title">Over time</p>', unsafe_allow_html=True)
    assist = kb_is_live()
    cols = st.columns(min(4, len(scans)))
    for i, s in enumerate(scans):
        with cols[i % len(cols)]:
            st.image(str(store.root / s.image_path), use_container_width=True)
            st.caption(s.taken_at[:10])
            render_verdict_pill(verdict_for_saved_scan(s))
            # Questions about a saved scan are the same questions as after a
            # live one — a participant reviewing an old result has exactly the
            # "what does this mean" problem the assistant exists for. Pinned
            # per scan rather than per case: a case can hold several scans with
            # different verdicts, and answering for the wrong one is the
            # failure this avoids.
            if assist and st.button(
                "Ask about this", key=f"case_ask_{s.id}", use_container_width=True
            ):
                ask_about_saved_scan(s)

    if len(scans) >= 2:
        with st.expander("Compare two scans"):
            labels = [f"{s.taken_at[:10]}" for s in scans]
            d_vals, b_vals, a_vals = [], [], []
            for s in scans:
                ab = json.loads(s.abcd_json)
                for letter, out in (("D", d_vals), ("B", b_vals), ("A", a_vals)):
                    dv = ab.get(letter, {}).get("value")
                    out.append(float(dv) if isinstance(dv, (int, float)) else 0.0)
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
            st.download_button(
                "Export CSV", store.export_case_csv(cid), file_name="case.csv", key="case_csv"
            )

    with actions_slot():
        add, back = st.columns([3, 2], gap="small")
        with add:
            if st.button(
                "Add a check to this spot", type="primary", key="case_add", use_container_width=True
            ):
                st.session_state["pending_case_id"] = cid
                navigate("camera")
        with back:
            if st.button("All saved spots", key="case_back", use_container_width=True):
                navigate("folder", selected_folder_id=fid) if fid else navigate("history")
