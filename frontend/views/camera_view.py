"""Take the photo — and say whether it is good enough *before* scanning it.

The old flow accepted any photo, ran the whole pipeline, then came back with
"image not clear" and no result. Here the three things that decide readability —
light, focus, and whether there is skin in frame — are shown as soon as a photo
exists, so a bad shot is retaken in two seconds instead of after a full scan.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from components.aperture import render_aperture, render_live_aperture
from components.app_bar import render_disclaimer_footer
from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link, render_primary_button
from navigation import navigate
from services.lesion_gate import has_skin
from services.quality import focus_meter, light_meter
from services.samples import list_sample_paths
from services.scan_flow import run_scan_and_store
from theme.tokens import TOKENS as T

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB — matches .streamlit/config.toml maxUploadSize

from services import pi_camera

_PICAMERA2_AVAILABLE = pi_camera.AVAILABLE


def _capture_picamera2() -> bytes | None:
    """Capture a settled, center-cropped JPEG from the shared Pi camera."""
    cam = pi_camera.get_camera()
    if cam is None:
        return None
    try:
        return cam.capture_still_jpeg()
    except Exception as exc:  # noqa: BLE001 — surfaced to user
        st.error(f"The camera did not take the photo: {exc}")
        return None


def _render_live_preview() -> None:
    """Live preview inside the circular field, so framing matches the result."""
    cam = pi_camera.get_camera()
    if cam is None:
        return
    render_live_aperture(pi_camera.preview_url())


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
        st.error(f"That picture is too big — the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        return None
    clean = _sanitize_upload(uploaded.getvalue())
    if clean is None:
        st.error("That file could not be opened. Use a JPEG or PNG picture.")
        return None
    return clean


def _meter(name: str, filled: int, value: str) -> str:
    bars = "".join(
        f'<span class="ds-meter-bar{" on" if i < filled else ""}"></span>' for i in range(3)
    )
    return (
        f'<div class="ds-meter"><span class="ds-meter-name">{name}</span>'
        f'<span class="ds-meter-bars">{bars}</span>'
        f'<span class="ds-meter-value">{value}</span></div>'
    )


def _render_photo_check(image_bytes: bytes) -> bool:
    """Show light / focus / skin readings for a captured photo.

    Returns True when the photo looks good enough to be worth scanning.
    """
    try:
        from backend.tflite_shared import decode_image_bytes_to_rgb

        rgb = decode_image_bytes_to_rgb(image_bytes)
    except Exception:  # noqa: BLE001 — never block capture on a preview check
        return True
    light_bars, light_word = light_meter(rgb)
    focus_bars, focus_word = focus_meter(rgb)
    # Asks the gate rather than repeating its threshold, so the meter can never
    # disagree with the check that actually blocks the scan.
    skin_ok = has_skin(rgb)
    st.markdown('<p class="ds-section-title">How this photo looks</p>', unsafe_allow_html=True)
    st.markdown(
        _meter("Light", light_bars, light_word)
        + _meter("Focus", focus_bars, focus_word)
        + _meter("Skin in frame", 3 if skin_ok else 0, "yes" if skin_ok else "not found"),
        unsafe_allow_html=True,
    )
    good = light_bars >= 2 and focus_bars >= 2 and skin_ok
    if not good:
        st.markdown(
            f'<p class="ds-reason">You can still check this photo, but taking a '
            f"better one will give a more reliable answer.</p>",
            unsafe_allow_html=True,
        )
    return good


def _store_capture(data: bytes | None) -> None:
    """Save new capture bytes and re-render so the photo check appears at once.

    The rerun matters: ``render_camera_view`` reads ``capture_image_bytes`` at
    the top, before these widgets run, so without it a freshly chosen photo
    would not show its light/focus/skin readings until the user touched
    something else. Guarded on the value actually changing, so it cannot loop.
    """
    if data is None or st.session_state.get("capture_image_bytes") == data:
        return
    st.session_state["capture_image_bytes"] = data
    st.rerun()


def _persist_capture(*, camera_key: str, upload_key: str) -> bytes | None:
    """Keep camera/upload bytes across Streamlit reruns (button clicks clear widget state)."""
    shot = st.camera_input("Camera", key=camera_key, label_visibility="collapsed")
    if shot is not None:
        _store_capture(shot.getvalue())
    up = st.file_uploader(
        "Or choose a picture",
        type=["jpg", "jpeg", "png"],
        key=upload_key,
        label_visibility="collapsed",
    )
    if up is not None:
        _store_capture(_accept_upload(up))
    return st.session_state.get("capture_image_bytes")


def render_camera_view(*, root: Path, backend, kind: str) -> None:
    with mobile_frame():
        if render_back_link("Back", key="cam_back"):
            st.session_state.pop("capture_image_bytes", None)
            navigate("home")
        pixels = float(st.session_state.get("pixels_per_mm_ui", 10.0))
        keras = str(st.session_state.get("SKIN_KERAS_PATH_UI", ""))
        strict_q = bool(st.session_state.get("strict_quality_gate", False))
        case_id = st.session_state.get("pending_case_id")
        image_bytes: bytes | None = None
        samples: list = []

        # Use the Pi hardware camera (picamera2) whenever it's importable — this runs on the
        # Pi itself, where the browser cannot reach the CSI camera. Independent of backend kind.
        use_hw_camera = _PICAMERA2_AVAILABLE and kind in ("pi", "local")
        captured = st.session_state.get("capture_image_bytes")

        left, right = st.columns([5, 6], gap="large")

        with left:
            if captured:
                # Rendered through the aperture component, which embeds the image
                # in its SVG. A bare <div> wrapper would not work: Streamlit
                # closes each st.markdown block, so the image would land outside.
                render_aperture(image=captured)
            elif use_hw_camera:
                _render_live_preview()
            else:
                render_aperture(hint="Put the spot inside the ring")

        with right:
            if captured:
                image_bytes = captured
                _render_photo_check(captured)
                st.markdown('<div class="ds-gap"></div>', unsafe_allow_html=True)
                if st.button("Take a different photo", key="cam_retake", use_container_width=True):
                    st.session_state.pop("capture_image_bytes", None)
                    st.rerun()
            elif use_hw_camera:
                st.markdown(
                    f'<p style="font-size:{T.font_sm}px;color:{T.text_muted};margin:0 0 {T.space_12}px">'
                    "Rest the camera on the skin so the spot sits inside the ring, "
                    "then hold still.</p>",
                    unsafe_allow_html=True,
                )
                if st.button("Take the photo", type="primary", key="pi_capture", use_container_width=True):
                    with st.spinner("Taking the photo…"):
                        shot = _capture_picamera2()
                    if shot is not None:
                        st.session_state["capture_image_bytes"] = shot
                        st.rerun()
                up = st.file_uploader(
                    "Or choose a picture",
                    type=["jpg", "jpeg", "png"],
                    key="hw_upload",
                    label_visibility="collapsed",
                )
                if up is not None:
                    _store_capture(_accept_upload(up))
            elif kind == "pi":
                st.info("Press Check this spot to take a photo with the Pi camera.")
                pu = st.file_uploader(
                    "Or send a picture to the Pi",
                    type=["jpg", "jpeg", "png"],
                    key="pi_upload",
                    label_visibility="collapsed",
                )
                if pu is not None:
                    _store_capture(_accept_upload(pu))
            else:
                st.markdown(
                    f'<p style="font-size:{T.font_sm}px;color:{T.text_muted};margin:0 0 {T.space_8}px">'
                    "Use the camera below, or choose a picture from this device.</p>",
                    unsafe_allow_html=True,
                )
                image_bytes = _persist_capture(camera_key="local_camera", upload_key="cam_upload")
                if kind == "mock":
                    samples = list_sample_paths(root)
                    if samples and image_bytes is None:
                        pick = st.selectbox("Or try an example", [l for l, _ in samples], key="mock_sample_pick")
                        chosen = dict(samples).get(pick)
                        if chosen:
                            st.image(str(chosen), width="stretch")

            st.markdown('<div class="ds-gap"></div>', unsafe_allow_html=True)
            if render_primary_button("Check this spot", key="cam_scan"):
                if kind == "mock" and image_bytes is None and samples:
                    pick = st.session_state.get("mock_sample_pick")
                    p = dict(samples).get(pick) if pick else None
                    if p:
                        image_bytes = p.read_bytes()
                if use_hw_camera and image_bytes is None:
                    st.error("Take the photo first.")
                elif kind in ("mock", "local") and image_bytes is None:
                    st.error("Take a photo, choose a picture, or try an example first.")
                elif kind == "pi" and image_bytes is None:
                    _run(backend, None, pixels, strict_q, keras, case_id)
                else:
                    _run(backend, image_bytes, pixels, strict_q, keras, case_id)

        render_disclaimer_footer()


def _run(backend, image_bytes, pixels, strict_q, keras, case_id) -> None:
    with st.spinner("Checking the spot…"):
        pl = run_scan_and_store(
            backend,
            image_bytes,
            pixels_per_mm=pixels,
            strict_quality=strict_q,
            keras_path=keras,
            case_id=str(case_id) if case_id else None,
        )
    st.session_state.pop("capture_image_bytes", None)
    st.session_state["last_result"] = pl
    navigate("results")
