from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from backend.contracts import ScanResult
from components.abcde_row import render_abcde_row
from components.app_bar import render_disclaimer_footer
from components.detection_headline import render_detection_headline
from components.image_compare import render_image_compare
from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link
from components.prob_bars import render_seven_class_expander, render_three_class_probs
from components.recommendation_card import render_recommendation_card
from components.urgency_pill import render_urgency_from_band
from navigation import navigate
from services.storage import get_storage


@st.dialog("Save to case")
def _save_dialog(model_path: str) -> None:
    store = get_storage()
    folders = store.list_folders()
    new_f = st.text_input("New folder name (optional)")
    folder_id = None
    if folders:
        folder_id = st.selectbox("Folder", [f.id for f in folders], format_func=lambda i: next(f.name for f in folders if f.id == i))
    name = st.text_input("Case name", "Lesion scan")
    site = st.selectbox("Body site", ["arm", "leg", "trunk", "face", "scalp", "hand", "foot", "other"])
    if st.button("Save", type="primary"):
        pl = st.session_state.get("last_result")
        if not pl or not pl.get("scan_result"):
            st.error("Nothing to save.")
            return
        if new_f.strip():
            folder_id = store.create_folder(new_f.strip()).id
        elif not folder_id:
            folder_id = store.create_folder("My scans").id
        case = store.create_case(folder_id, name.strip() or "Untitled", body_site=site)
        sr = pl["scan_result"]
        import cv2

        img = sr.image_jpg_bytes
        if pl.get("rgb") is not None:
            bgr = cv2.cvtColor(pl["rgb"], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                img = buf.tobytes()
        saved = store.save_scan(case.id, pl, img, model_path=model_path)
        if pl.get("abcde") and saved.e_json:
            import json

            ab = dict(pl["abcde"])
            ab["E"] = json.loads(saved.e_json)
            pl = dict(pl)
            pl["abcde"] = ab
            st.session_state["last_result"] = pl
        navigate("case", selected_case_id=case.id, selected_folder_id=folder_id)


def render_results_view(*, root: Path, model_path: str) -> None:
    with mobile_frame():
        pl = st.session_state.get("last_result")
        if not pl:
            st.info("Run a scan from Home first.")
            if render_back_link("Home", key="res_empty_home"):
                navigate("home")
            return
        if pl.get("blocked") and not pl.get("scan_result"):
            st.error("Quality checks failed.")
            for r in pl.get("quality", {}).get("reasons", []):
                st.warning(r)
            return
        if pl.get("error") and not pl.get("scan_result"):
            st.error(pl["error"])
            return
        render_urgency_from_band(str(pl.get("risk_band", "low")))
        sr = pl.get("scan_result")
        if isinstance(sr, ScanResult):
            render_detection_headline(sr.label, float(sr.confidence))
            t_path = Path(model_path).resolve().parent / "temperature.json"
            if not t_path.is_file():
                t_path = root / "models" / "temperature.json"
            if t_path.is_file():
                t_val = json.loads(t_path.read_text()).get("T", 1.0)
                st.caption(f"ℹ Calibrated confidence (temperature T = {t_val:.2f})")
            render_image_compare(
                pl.get("rgb"),
                pl.get("gradcam_overlay_jpg"),
                gradcam_caption="Grad-CAM unavailable — set SKIN_KERAS_PATH in Settings",
            )
            if pl.get("rgb_before") is not None and pl.get("rgb") is not None:
                with st.expander("Preprocessing debug (before / after)"):
                    c1, c2 = st.columns(2)
                    c1.image(pl["rgb_before"], caption="Before", use_container_width=True)
                    c2.image(pl["rgb"], caption="After", use_container_width=True)
            render_abcde_row(pl.get("abcde"))
            render_three_class_probs(sr.probs)
            render_seven_class_expander(pl.get("seven_class_probs"), str(st.session_state.get("SKIN_KERAS_PATH_UI", "")))
            render_recommendation_card(sr.action)
        qd = pl.get("quality", {}).get("reason_details", [])
        for _code, label, _sev in qd:
            st.warning(label)
        for w in pl.get("quality_warnings", []):
            st.warning(w)
        if pl.get("trust_line"):
            st.caption(pl["trust_line"])
        render_disclaimer_footer()
        c1, c2, c3 = st.columns(3)
        with c1:
            if render_back_link("Home", key="res_home"):
                navigate("home")
        with c2:
            if st.button("Save to case…", key="res_save"):
                _save_dialog(model_path)
        with c3:
            if st.button("Clear", key="res_clear"):
                st.session_state.pop("last_result", None)
                st.rerun()
