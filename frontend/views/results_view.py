"""The result: what to do next, and what the scan actually saw.

The photo, its verdict ring and the five ABCDE marks live in the instrument
band beside this screen, so the page band is free to be one column of prose:
headline, why, the sentence addressed to a person, then the three findings in
plain words and the panel that says what to do about them.

Everything technical moved to the ``staff`` route. It used to be an expander
here, open to anyone, while the saved photos next door sat behind a passcode.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from backend.assistant import kb_is_live
from backend.contracts import ScanResult
from components.actions import actions_slot
from components.instrument import render_head
from components.verdict_card import render_verdict_card, render_verdict_note
from navigation import navigate
from services.format import sign_ink, tone_chip, tone_colors
from services.kiosk import is_kiosk
from services.storage import get_storage
from services.verdict import resolve_verdict, retake_verdict, scan_signs
from views.assistant_view import ask_about_last_result


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
    st.session_state.pop("capture_image_bytes", None)
    navigate("home")


def _render_signs(abcde: dict | None) -> None:
    """The three findings, in words a visitor can check against their own skin."""
    signs = scan_signs(abcde)
    if not signs:
        st.markdown(
            '<div class="ds-mid"><p class="ds-reason">The spot could not be outlined, '
            "so the five checks were skipped.</p></div>",
            unsafe_allow_html=True,
        )
        return
    rows = "".join(
        f'<div class="ds-sign"><span class="ds-sign-mark" style="background:{sign_ink(s.tier)}">'
        f'</span><span class="ds-row-grow">{s.text}</span></div>'
        for s in signs
    )
    st.markdown(
        '<div class="ds-mid"><p class="ds-section-title">What the scan saw</p>'
        f"{rows}</div>",
        unsafe_allow_html=True,
    )


def _render_stopped(pl: dict) -> None:
    """No usable scan: the photo could not be read, or holds no skin spot."""
    v = pl.get("verdict") or retake_verdict(pl.get("quality"))
    ink, fill = tone_colors(v.tone)
    chip = (
        f'<span class="ds-pill" style="background:{fill};color:{ink}">'
        f"{tone_chip(v.tone)}</span>"
    )
    render_head("Result · nothing was read", "", extra=chip)
    render_verdict_card(v)
    render_verdict_note(v)

    with actions_slot():
        first, second = st.columns([3, 2], gap="small")
        with first:
            if st.button(
                "Take another photo", type="primary", key="stop_retry", use_container_width=True
            ):
                st.session_state.pop("last_result", None)
                st.session_state.pop("capture_image_bytes", None)
                navigate("camera")
        with second:
            # A refusal must not be a dead end. Taking a better photo is the
            # right first answer, but when the scanner is simply wrong about
            # the photo — it does happen — the person holding the lesion needs
            # a way through.
            #
            # Except when the refusal was about the *subject* rather than the
            # photo. "This is not one spot" and "this is not something it can
            # read" are hard: the classifier only knows benign, pre-cancerous
            # and malignant, so forcing a face through does not produce a
            # cautious answer, it produces a confident wrong one. Every refusal
            # about framing, focus or light stays overridable — those are the
            # ones that could be wrong about a real lesion.
            check = pl.get("frame_check")
            can_override = getattr(check, "can_override", True)
            if can_override and st.session_state.get("capture_image_bytes"):
                if st.button("Check it anyway", key="stop_force", use_container_width=True):
                    st.session_state["force_rescan"] = True
                    st.session_state.pop("last_result", None)
                    navigate("reading")
            else:
                # Reached when the bytes are gone, and now also when the refusal
                # is hard. The nav keys are the way out either way, but a screen
                # that offers no exit of its own is a dead end on a kiosk.
                if st.button("Back to start", key="stop_home", use_container_width=True):
                    _go_home()


def render_results_view(*, root: Path, model_path: str) -> None:  # noqa: ARG001 — root used by staff view
    pl = st.session_state.get("last_result")
    if not pl:
        render_head("Result", "No scan yet", "Take a photo to see a result here.")
        with actions_slot():
            if st.button(
                "Start a skin check", type="primary", key="res_empty_home", use_container_width=True
            ):
                navigate("camera")
        return
    if pl.get("blocked") and not pl.get("scan_result"):
        _render_stopped(pl)
        return
    if pl.get("error") and not pl.get("scan_result"):
        render_head("Result", "That scan did not finish", pl["error"])
        with actions_slot():
            if st.button("Back to start", type="primary", key="res_err_home", use_container_width=True):
                _go_home()
        return

    sr = pl.get("scan_result")
    if not isinstance(sr, ScanResult):
        render_head("Result", "That scan did not finish", "Please take another photo.")
        with actions_slot():
            if st.button("Back to start", type="primary", key="res_bad_home", use_container_width=True):
                _go_home()
        return

    # The verdict is the ONLY risk-language element on this screen — every
    # message comes from services.verdict.
    v = pl.get("verdict") or resolve_verdict(sr, pl.get("quality"))
    ink, fill = tone_colors(v.tone)
    chip = (
        f'<span class="ds-pill" style="background:{fill};color:{ink}">{tone_chip(v.tone)}</span>'
    )
    render_head("Result · what to do next", "", extra=chip)
    render_verdict_card(v)
    _render_signs(pl.get("abcde"))
    render_verdict_note(v)

    # A forced scan bypassed the checks that would have refused this photo, so
    # the reason it was refused has to travel with the result. Silently showing
    # a normal-looking verdict for a photo the scanner could not read is exactly
    # the false confidence this app is built to avoid.
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

    with actions_slot():
        done, save = st.columns(2, gap="small")
        with done:
            if st.button("Done", type="primary", key="res_home", use_container_width=True):
                _go_home()
        with save:
            if st.button("Save this scan", key="res_save", use_container_width=True):
                _save_dialog(model_path)
        ask, staff = st.columns(2, gap="small")
        with ask:
            if kb_is_live() and st.button(
                "Ask a question about this", key="res_ask", use_container_width=True
            ):
                ask_about_last_result()
        with staff:
            if st.button("Details for staff →", key="res_staff", use_container_width=True):
                navigate("staff")
