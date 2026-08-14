"""The result: one photo, one verdict, side by side.

Landscape two-pane because the kiosk panel is 1024x600 — stacking the photo
above the verdict pushes the answer below the fold on a screen nobody scrolls.
The aperture on the left carries the ABCDE marks on its rim, so the lesion and
its five measurements read as one object instead of two competing blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from backend.assistant import kb_is_live
from backend.contracts import ScanResult
from components.abcde_row import render_abcde_row
from components.aperture import render_aperture
from components.app_bar import render_disclaimer_footer
from components.image_compare import render_image_compare
from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link
from components.prob_bars import render_seven_class_bars, render_three_class_probs
from components.verdict_card import render_verdict_card
from navigation import navigate
from services.eigencam import cam_model_path
from views.assistant_view import ask_about_last_result
from services.kiosk import is_kiosk
from services.pipeline import trust_line
from services.scan_flow import build_attention_overlay
from services.storage import get_storage
from services.verdict import resolve_verdict, retake_verdict
from theme.tokens import TOKENS as T


def cam_available() -> bool:
    """Whether an attention view could be built at all on this device."""
    return cam_model_path().is_file()


@st.dialog("Save this scan")
def _save_dialog(model_path: str) -> None:
    from datetime import date

    kiosk = is_kiosk()
    store = get_storage()
    folders = store.list_folders()
    folder_id = None
    if kiosk:
        # Keyboard-less kiosk: presets only, no free-text typing.
        new_f = ""
        if folders:
            folder_id = st.selectbox("Folder", [f.id for f in folders], format_func=lambda i: next(f.name for f in folders if f.id == i))
        site = st.selectbox("Where on the body?", ["arm", "leg", "trunk", "face", "scalp", "hand", "foot", "other"])
        name = f"{site} {date.today().isoformat()}"
        st.caption(f"This will be saved as: {name}")
    else:
        new_f = st.text_input("New folder name (optional)")
        if folders:
            folder_id = st.selectbox("Folder", [f.id for f in folders], format_func=lambda i: next(f.name for f in folders if f.id == i))
        name = st.text_input("Name for this spot", "Lesion scan")
        site = st.selectbox("Where on the body?", ["arm", "leg", "trunk", "face", "scalp", "hand", "foot", "other"])
    consent = st.checkbox(
        "I agree to keep this photo of my skin on this device so it can be "
        "compared at a later visit. Staff can delete it at any time.",
        key="save_consent",
    )
    if st.button("Save", type="primary", disabled=not consent):
        pl = st.session_state.get("last_result")
        if not pl or not pl.get("scan_result"):
            st.error("There is no scan to save.")
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
        saved = store.save_scan(case.id, pl, img, model_path=model_path, consent=True)
        if pl.get("abcde") and saved.e_json:
            ab = dict(pl["abcde"])
            ab["E"] = json.loads(saved.e_json)
            pl = dict(pl)
            pl["abcde"] = ab
            st.session_state["last_result"] = pl
        navigate("case", selected_case_id=case.id, selected_folder_id=folder_id)


def _go_home() -> None:
    """Leaving Results discards the photo — docs/PRIVACY.md promises this.

    Navigating away without clearing left the previous participant's skin in
    session state, one Back tap from the next person.
    """
    st.session_state.pop("last_result", None)
    navigate("home")


def _render_abcde_legend(abcde: dict | None) -> None:
    """Say what the five marks around the ring mean.

    Without this the rim is five coloured arcs a visitor cannot read. The
    wording stays plain — "the five things doctors look at" rather than the
    asymmetry/border/colour/diameter/evolving vocabulary, which is in the staff
    section where it belongs.
    """
    if not abcde:
        st.markdown(
            f'<p style="text-align:center;font-size:{T.font_2xs}px;color:{T.text_muted};'
            f'margin-top:{T.space_8}px">The spot could not be outlined, so the five '
            "checks were skipped.</p>",
            unsafe_allow_html=True,
        )
        return
    keys = (("Normal", T.text_muted), ("Borderline", T.melanin), ("Stands out", T.erythema))
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px">'
        f'<span style="width:12px;height:3px;border-radius:2px;background:{ink};'
        'display:inline-block"></span>'
        f"<span>{label}</span></span>"
        for label, ink in keys
    )
    st.markdown(
        f'<p style="text-align:center;font-size:{T.font_2xs}px;color:{T.text_muted};'
        f'letter-spacing:.1em;text-transform:uppercase;margin:{T.space_12}px 0 {T.space_4}px">'
        "A B C D E · the five things doctors look at</p>"
        f'<div style="display:flex;justify-content:center;gap:{T.space_12}px;'
        f'font-size:{T.font_2xs}px;color:{T.text_muted};margin-bottom:{T.space_16}px">'
        f"{swatches}</div>",
        unsafe_allow_html=True,
    )


def _render_stopped(pl: dict) -> None:
    """No usable scan: the photo could not be read, or holds no skin spot."""
    v = pl.get("verdict") or retake_verdict(pl.get("quality"))
    left, right = st.columns([5, 6], gap="large")
    with left:
        render_aperture(image=pl.get("rgb"), verdict=v, dashed=True)
    with right:
        render_verdict_card(v)
        st.markdown('<div class="ds-gap"></div>', unsafe_allow_html=True)
        if st.button("Take another photo", type="primary", key="stop_retry", use_container_width=True):
            st.session_state.pop("last_result", None)
            navigate("camera")
        # A refusal must not be a dead end. Taking a better photo is the right
        # first answer, but when the scanner is simply wrong about the photo —
        # it does happen — the person holding the lesion needs a way through.
        if st.session_state.get("capture_image_bytes") and st.button(
            "Check it anyway", key="stop_force", use_container_width=True
        ):
            st.session_state["force_rescan"] = True
            st.session_state.pop("last_result", None)
            navigate("camera")
        if st.button("Back to start", key="stop_home", use_container_width=True):
            _go_home()
    render_disclaimer_footer()


def render_results_view(*, root: Path, model_path: str) -> None:
    with mobile_frame():
        pl = st.session_state.get("last_result")
        if not pl:
            st.info("Take a photo first.")
            if render_back_link("Back to start", key="res_empty_home"):
                navigate("home")
            return
        if pl.get("blocked") and not pl.get("scan_result"):
            _render_stopped(pl)
            return
        if pl.get("error") and not pl.get("scan_result"):
            st.error(pl["error"])
            if st.button("Back to start", key="res_err_home", use_container_width=True):
                _go_home()
            return

        sr = pl.get("scan_result")
        if not isinstance(sr, ScanResult):
            st.error("That scan did not finish. Please take another photo.")
            if st.button("Back to start", key="res_bad_home", use_container_width=True):
                _go_home()
            return

        v = pl.get("verdict") or resolve_verdict(sr, pl.get("quality"))
        left, right = st.columns([5, 6], gap="large")
        with left:
            # The verdict card is the ONLY risk-language element on this screen —
            # every message comes from services.verdict.
            render_aperture(image=pl.get("rgb"), verdict=v, abcde=pl.get("abcde"))
            _render_abcde_legend(pl.get("abcde"))
        with right:
            render_verdict_card(v)
            st.markdown('<div class="ds-gap"></div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Save this scan", key="res_save", use_container_width=True):
                    _save_dialog(model_path)
            with b2:
                if st.button("Done", type="primary", key="res_home", use_container_width=True):
                    _go_home()
            if kb_is_live() and st.button("Ask a question about this", key="res_ask", use_container_width=True):
                ask_about_last_result()

        # A forced scan bypassed the checks that would have refused this photo,
        # so the reason it was refused has to travel with the result. Silently
        # showing a normal-looking verdict for a photo the scanner could not
        # read is exactly the false confidence this app is built to avoid.
        if pl.get("forced"):
            fc = pl.get("frame_check")
            why = "; ".join(getattr(fc, "reasons", ()) or ()) if fc else ""
            st.warning(
                "This photo did not pass the usual checks"
                + (f" ({why})" if why else "")
                + " and was read anyway. Treat the result with extra caution."
            )

        for _code, label, _sev in pl.get("quality", {}).get("reason_details", []):
            st.warning(label)

        # Everything technical lives in ONE collapsed section — the screen above
        # stays in plain language for visitors and health workers.
        with st.expander("Details for staff"):
            render_abcde_row(pl.get("abcde"), show_values=True)
            # The heatmap is built on request, not on every scan: it costs a
            # second full model pass plus a full-resolution blend, and this
            # panel is collapsed by default. st.expander is not lazy — its body
            # runs on every rerun — so the button is what defers the work.
            if pl.get("attention_overlay_jpg"):
                render_image_compare(
                    pl.get("rgb"), pl.get("attention_overlay_jpg"), note=pl.get("attention_note")
                )
            elif not cam_available():
                st.caption("The attention view is not available on this device.")
            elif st.button("Show where the scanner looked", key="res_attention"):
                with st.spinner("Building the attention view…"):
                    build_attention_overlay(pl, str(st.session_state.get("SKIN_KERAS_PATH_UI", "")))
                st.session_state["last_result"] = pl
                st.rerun()
            render_three_class_probs(sr.probs)
            render_seven_class_bars(pl.get("seven_class_probs"))
            line = trust_line(pl)
            if line:
                st.caption(line)
            t_path = Path(model_path).resolve().parent / "temperature.json"
            if not t_path.is_file():
                t_path = root / "models" / "temperature.json"
            if t_path.is_file():
                t_val = json.loads(t_path.read_text()).get("T", 1.0)
                if t_val and float(t_val) != 1.0:
                    # Displayed confidence IS calibrated (backend/tflite_shared.apply_temperature);
                    # the screening decision itself still uses raw probabilities.
                    st.caption(f"Confidence calibrated with temperature scaling (T = {float(t_val):.2f})")
            if pl.get("rgb_before") is not None and pl.get("rgb_analysis") is not None:
                st.markdown("**Preprocessing (before / after — ABCDE path only)**")
                c1, c2 = st.columns(2)
                c1.image(pl["rgb_before"], caption="Original (classifier input)", width="stretch")
                c2.image(pl["rgb_analysis"], caption="Enhanced (ABCDE/segmentation)", width="stretch")

        render_disclaimer_footer()
