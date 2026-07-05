from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from backend.contracts import ScanResult
from components.abcde_row import render_abcde_row
from components.app_bar import render_disclaimer_footer
from components.image_compare import render_image_compare
from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link
from components.prob_bars import render_seven_class_bars, render_three_class_probs
from components.recommendation_card import render_recommendation_card
from components.risk_ring import render_risk_ring
from navigation import navigate
from services.storage import get_storage

# Below this calibrated top-class confidence (%), the model isn't sure enough to
# stand behind a single label — show an "inconclusive" banner instead of pretending.
# Tunable via env so it can be adjusted without a code change.
try:
    _INCONCLUSIVE_BELOW = float(os.environ.get("SKIN_INCONCLUSIVE_BELOW", "45"))
except ValueError:
    _INCONCLUSIVE_BELOW = 45.0


@st.dialog("Save to case")
def _save_dialog(model_path: str) -> None:
    import os
    from datetime import date

    kiosk = os.environ.get("SKIN_KIOSK") == "1"
    store = get_storage()
    folders = store.list_folders()
    folder_id = None
    if kiosk:
        # Keyboard-less kiosk: presets only, no free-text typing.
        new_f = ""
        if folders:
            folder_id = st.selectbox("Folder", [f.id for f in folders], format_func=lambda i: next(f.name for f in folders if f.id == i))
        site = st.selectbox("Body site", ["arm", "leg", "trunk", "face", "scalp", "hand", "foot", "other"])
        name = f"{site} {date.today().isoformat()}"
        st.caption(f"Case name: {name}")
    else:
        new_f = st.text_input("New folder name (optional)")
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
            st.error("Photo quality was too low to analyze — please retake.")
            for r in pl.get("quality", {}).get("reasons", []):
                st.warning(r)
            return
        if pl.get("error") and not pl.get("scan_result"):
            st.error(pl["error"])
            return
        sr = pl.get("scan_result")
        if isinstance(sr, ScanResult):
            render_risk_ring(sr.label, float(sr.confidence), str(pl.get("risk_band", "low")))
            # Honest "neither / not sure" output: if even the calibrated top class is
            # weak, tell the user it's inconclusive rather than over-trusting the label.
            if float(sr.confidence) < _INCONCLUSIVE_BELOW:
                st.warning(
                    "⚠ Inconclusive — the model is not confident about this result. "
                    "Retake with better lighting/focus, or consult a health professional."
                )
            render_image_compare(pl.get("rgb"), pl.get("gradcam_overlay_jpg"))
            if pl.get("borderline_note"):
                st.warning(pl["borderline_note"])
            render_abcde_row(pl.get("abcde"))
            render_three_class_probs(sr.probs)
            render_recommendation_card(sr.action, str(pl.get("risk_band", "low")))
        qd = pl.get("quality", {}).get("reason_details", [])
        for _code, label, _sev in qd:
            st.warning(label)
        for w in pl.get("quality_warnings", []):
            st.warning(w)
        # Everything technical lives in ONE collapsed expander — the main view
        # stays in plain language for ordinary users and health workers.
        if isinstance(sr, ScanResult):
            with st.expander("🔧 Technical details"):
                if pl.get("trust_line"):
                    st.caption(pl["trust_line"])
                t_path = Path(model_path).resolve().parent / "temperature.json"
                if not t_path.is_file():
                    t_path = root / "models" / "temperature.json"
                if t_path.is_file():
                    t_val = json.loads(t_path.read_text()).get("T", 1.0)
                    if t_val and float(t_val) != 1.0:
                        # Displayed confidence IS calibrated (backend/tflite_shared.apply_temperature);
                        # the screening decision itself still uses raw probabilities.
                        st.caption(f"Confidence calibrated with temperature scaling (T = {float(t_val):.2f})")
                if pl.get("abcde"):
                    render_abcde_row(pl.get("abcde"), show_values=True)
                render_seven_class_bars(pl.get("seven_class_probs"))
                if not pl.get("gradcam_overlay_jpg"):
                    st.caption("AI attention heatmap (Grad-CAM) unavailable — the full Keras model is not loaded on this device.")
                if pl.get("rgb_before") is not None and pl.get("rgb_analysis") is not None:
                    st.markdown("**Preprocessing debug (before / after — ABCDE path only)**")
                    c1, c2 = st.columns(2)
                    c1.image(pl["rgb_before"], caption="Original (CNN input)", width="stretch")
                    c2.image(pl["rgb_analysis"], caption="Enhanced (ABCDE/segmentation)", width="stretch")
        render_disclaimer_footer()
        # 2x2 grid — larger touch targets than a crushed 3-column row.
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💬 Ask about this result", key="res_ask", use_container_width=True):
                navigate("assistant")
        with c2:
            if st.button("Save to case…", key="res_save", use_container_width=True):
                _save_dialog(model_path)
        c3, c4 = st.columns(2)
        with c3:
            if render_back_link("Home", key="res_home"):
                navigate("home")
        with c4:
            if st.button("Clear", key="res_clear", use_container_width=True):
                st.session_state.pop("last_result", None)
                st.rerun()
