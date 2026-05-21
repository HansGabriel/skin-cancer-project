from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from components.app_bar import render_disclaimer_footer
from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link, render_primary_button
from components.viewfinder import viewfinder_slot
from navigation import navigate
from services.scan_flow import run_scan_and_store
from views.scan_view import list_sample_paths

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB — matches .streamlit/config.toml maxUploadSize


def _sanitize_upload(raw: bytes) -> bytes | None:
    """Re-encode user-supplied image bytes to JPEG, stripping EXIF and other metadata."""
    try:
        with Image.open(io.BytesIO(raw)) as img:
            rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:  # noqa: BLE001 — surfaced to user as a friendly error
        return None


def _accept_upload(uploaded) -> bytes | None:
    if uploaded is None:
        return None
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error(f"Image too large — max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        return None
    clean = _sanitize_upload(uploaded.getvalue())
    if clean is None:
        st.error("Could not read that image. Use a JPEG or PNG.")
        return None
    return clean


def _persist_capture(*, camera_key: str, upload_key: str) -> bytes | None:
    """Keep camera/upload bytes across Streamlit reruns (button clicks clear widget state)."""
    shot = st.camera_input("Camera", key=camera_key, label_visibility="collapsed")
    if shot is not None:
        st.session_state["capture_image_bytes"] = shot.getvalue()
    up = st.file_uploader(
        "Or upload a photo",
        type=["jpg", "jpeg", "png"],
        key=upload_key,
        label_visibility="collapsed",
    )
    if up is not None:
        cleaned = _accept_upload(up)
        if cleaned is not None:
            st.session_state["capture_image_bytes"] = cleaned
    raw = st.session_state.get("capture_image_bytes")
    if raw:
        st.image(raw, width="stretch")
        if st.button("Clear photo", key=f"clear_{camera_key}"):
            st.session_state.pop("capture_image_bytes", None)
            st.rerun()
    return raw


def render_camera_view(*, root: Path, backend, kind: str) -> None:
    with mobile_frame():
        if render_back_link("back", key="cam_back"):
            st.session_state.pop("capture_image_bytes", None)
            navigate("home")
        pixels = float(st.session_state.get("pixels_per_mm_ui", 10.0))
        keras = str(st.session_state.get("SKIN_KERAS_PATH_UI", ""))
        strict_q = bool(st.session_state.get("strict_quality_gate", True))
        case_id = st.session_state.get("pending_case_id")
        image_bytes: bytes | None = None
        samples: list = []
        scanning = st.session_state.get("_cam_scanning", False)

        if kind == "pi":
            st.caption("Pi camera — inference runs on the Raspberry Pi.")
        else:
            st.caption(
                "Use your device webcam below, or upload a photo. "
                "(WSL/Linux: if the camera is blank, upload instead — the browser needs camera permission.)"
            )

        with viewfinder_slot(shutter=scanning):
            if kind == "pi":
                st.info("Press START SCAN to capture from the Pi camera over the network.")
                pu = st.file_uploader(
                    "Or send a test image to the Pi",
                    type=["jpg", "jpeg", "png"],
                    key="pi_upload",
                    label_visibility="collapsed",
                )
                if pu is not None:
                    cleaned = _accept_upload(pu)
                    if cleaned is not None:
                        image_bytes = cleaned
                        st.session_state["capture_image_bytes"] = cleaned
                        st.image(cleaned, width="stretch")
            elif kind in ("mock", "local"):
                image_bytes = _persist_capture(camera_key="local_camera", upload_key="cam_upload")
                if kind == "mock":
                    samples = list_sample_paths(root)
                    if samples and image_bytes is None:
                        pick = st.selectbox("Or pick a sample", [l for l, _ in samples], key="mock_sample_pick")
                        chosen = dict(samples).get(pick)
                        if chosen:
                            st.image(str(chosen), width="stretch")
            else:
                image_bytes = _persist_capture(camera_key="local_camera_alt", upload_key="cam_upload_alt")

        if render_primary_button("START SCAN", key="cam_scan"):
            if kind == "mock" and image_bytes is None and samples:
                pick = st.session_state.get("mock_sample_pick")
                p = dict(samples).get(pick) if pick else None
                if p:
                    image_bytes = p.read_bytes()
            if kind in ("mock", "local") and image_bytes is None:
                st.error("Take a photo with the camera, upload an image, or pick a sample.")
            elif kind == "pi" and image_bytes is None:
                pass  # Pi capture without upload
            else:
                st.session_state["_cam_scanning"] = True
                with st.spinner("Analyzing…"):
                    pl = run_scan_and_store(
                        backend,
                        image_bytes if kind != "pi" or image_bytes else None,
                        pixels_per_mm=pixels,
                        strict_quality=strict_q,
                        keras_path=keras,
                        case_id=str(case_id) if case_id else None,
                    )
                st.session_state.pop("_cam_scanning", None)
                st.session_state.pop("capture_image_bytes", None)
                st.session_state["last_result"] = pl
                navigate("results")
        render_disclaimer_footer()
