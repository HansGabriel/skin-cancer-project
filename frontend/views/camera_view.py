"""Take the photo, then say whether it is good enough *before* scanning it.

Two steps of the three-step flow live here, switched on whether a photo exists
yet — the instrument band beside them shows the live preview or the capture, so
this file only draws the page band.

**Step 1 — frame.** How to hold the device.

**Step 2 — check the photo.** The three things that decide readability (light,
focus, and whether there is skin in frame) are shown as soon as a photo exists,
so a bad shot is retaken in two seconds instead of after a full scan.

Step 2 is also where an unclear photo is stopped. It is a *soft* gate: when the
readings are poor the primary action becomes "Take another photo" and going on
anyway is demoted to secondary. Deliberately not enforced in the pipeline —
the Settings "Check photos strictly" switch does that, and it was measured
refusing **72% of genuine HAM10000 lesions** (see
``tests/test_gate_real_images.py``). Steering beats
refusing when the cost of a wrong refusal is someone's melanoma.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import streamlit as st
from PIL import Image

from components.actions import actions_slot
from components.instrument import render_head
from navigation import navigate
from services import photo_cache
from services.lesion_gate import has_skin
from services.quality import focus_meter, light_meter
from services.samples import list_sample_paths
from theme.tokens import TOKENS as T

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB — matches .streamlit/config.toml maxUploadSize

# NOTE: the resolution cap deliberately does NOT live here. It used to, and
# that silently broke the thing it was paired with: services.pipeline caps at
# MAX_WORK_PX and compensates `pixels_per_mm` by the same factor, but a frame
# already shrunk to 1024 here fails its `max(shape) > MAX_WORK_PX` test, so the
# compensation never ran and every millimetre reading stayed wrong by the
# resize factor. One place shrinks the image, and it is the one place that also
# knows the scale.

from services import pi_camera

_PICAMERA2_AVAILABLE = pi_camera.AVAILABLE

_FRAMING_TIPS = (
    "Spot inside the ring, edges included",
    "Even light — no shadow, no glare",
    "Two seconds still, then tap",
)


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


def _sanitize_upload(raw: bytes) -> bytes | None:
    """Re-encode user-supplied image bytes to JPEG, stripping EXIF and metadata.

    Resolution is NOT capped here — see the note above ``MAX_UPLOAD_BYTES``.
    """
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


@lru_cache(maxsize=2)
def _photo_readings(image_bytes: bytes) -> tuple[str, bool]:
    """(meter markup, good enough) for a captured photo.

    Memoised on the bytes. Step 2 recomputes this on every rerun, and it costs a
    full JPEG decode plus a 512px Laplacian plus a skin-mask pass — on the screen
    people tap the most, on the slowest machine this runs on. The readings are a
    pure function of the photo, so the second and later reruns are free.

    Cleared when the photo leaves session state (``navigation._drop_photo_caches``)
    so no participant's capture outlives their session in this cache.
    """
    try:
        from backend.tflite_shared import decode_image_bytes_to_rgb

        rgb = decode_image_bytes_to_rgb(image_bytes)
    except Exception:  # noqa: BLE001 — never block capture on a preview check
        return "", True
    light_bars, light_word = light_meter(rgb)
    focus_bars, focus_word = focus_meter(rgb)
    # Asks the gate rather than repeating its threshold, so the meter can never
    # disagree with the check that actually blocks the scan.
    skin_ok = has_skin(rgb)
    markup = (
        '<div class="ds-mid">'
        + _meter("Light", light_bars, light_word)
        + _meter("Focus", focus_bars, focus_word)
        + _meter("Skin in frame", 3 if skin_ok else 0, "yes" if skin_ok else "not found")
        + "</div>"
    )
    return markup, bool(light_bars >= 2 and focus_bars >= 2 and skin_ok)


def forget_photo_readings() -> None:
    """Drop every cached reading. Called when the photo leaves session state."""
    _photo_readings.cache_clear()


photo_cache.register(forget_photo_readings)


def _discard_capture() -> None:
    """Throw the photo away and go back to step 1.

    These buttons never navigate — they drop the bytes and rerun into step 1 of
    the same screen — so the router's clearing never ran for them and the
    memoised copies survived a boundary docs/PRIVACY.md names by name.
    """
    st.session_state.pop("capture_image_bytes", None)
    st.session_state.pop("capture_from_device", None)
    photo_cache.forget_photos()
    st.rerun()


def _store_capture(data: bytes | None) -> None:
    """Save new capture bytes and re-render so the photo check appears at once.

    The rerun matters: the view reads ``capture_image_bytes`` at the top, before
    these widgets run, so without it a freshly chosen photo would not show its
    light/focus/skin readings until the user touched something else. Guarded on
    the value actually changing, so it cannot loop.
    """
    if data is None or st.session_state.get("capture_image_bytes") == data:
        return
    st.session_state["capture_image_bytes"] = data
    # Everything arriving here is an upload, a browser camera or a demo file:
    # photographed at an unknown distance through unknown optics, so its
    # real-world scale is not knowable. Only the device camera at its fixed
    # working distance sets this True (see the picamera2 path below).
    st.session_state["capture_from_device"] = False
    st.rerun()


def _render_step_two(kind: str) -> None:
    """The photo exists: show what it looks like and whether to go on."""
    captured = st.session_state["capture_image_bytes"]
    meters, good = _photo_readings(captured)
    if good:
        render_head(
            "Step 2 of 3 · check the photo",
            "This photo can be read",
            "All three readings are good enough for a reliable answer.",
        )
    else:
        render_head(
            "Step 2 of 3 · check the photo",
            "This photo is hard to read",
            "A better photo will give a more reliable answer. You can still go on.",
        )
    st.markdown(meters or '<div class="ds-mid"></div>', unsafe_allow_html=True)

    with actions_slot():
        first, second = st.columns([3, 2], gap="small")
        if good:
            with first:
                if st.button(
                    "Check this spot", type="primary", key="cam_scan", use_container_width=True
                ):
                    navigate("reading")
            with second:
                if st.button("Take another", key="cam_retake", use_container_width=True):
                    _discard_capture()
        else:
            # Order swapped, not the button removed: refusing outright is what
            # the strict gate does, and it refuses real lesions.
            with first:
                if st.button(
                    "Take another photo",
                    type="primary",
                    key="cam_retake_soft",
                    use_container_width=True,
                ):
                    _discard_capture()
            with second:
                if st.button("Check it anyway", key="cam_scan_soft", use_container_width=True):
                    navigate("reading")


def _render_step_one(root: Path, kind: str) -> None:
    """No photo yet: how to frame it, and the controls to take one."""
    use_hw_camera = _PICAMERA2_AVAILABLE and kind in ("pi", "local")

    # Remote Pi with no local camera: the device captures and classifies in one
    # call, so there is nothing to hold here and the scan starts with no bytes.
    # Without this branch the "reading" route is unreachable on that backend and
    # the Pi camera cannot be used from a PC session at all.
    if kind == "pi" and not use_hw_camera:
        render_head(
            "Step 1 of 3 · frame the spot",
            "Use the camera on the device",
            "The scanner takes the photo itself. Rest it on the skin so the spot "
            "sits in the middle of the ring, then start the check.",
        )
        pu = st.file_uploader(
            "Or send a picture to the scanner",
            type=["jpg", "jpeg", "png"],
            key="pi_upload",
            label_visibility="collapsed",
        )
        if pu is not None:
            _store_capture(_accept_upload(pu))
        with actions_slot():
            if st.button(
                "Check this spot", type="primary", key="pi_scan", use_container_width=True
            ):
                navigate("reading")
        return

    render_head(
        "Step 1 of 3 · frame the spot",
        "Fill the ring, then hold still",
        "Rest the camera on the skin so the spot sits in the middle of the ring.",
    )

    if use_hw_camera:
        tips = "".join(
            f'<div class="ds-row"><span class="ds-dot"></span>'
            f'<span class="ds-row-grow">{tip}</span></div>'
            for tip in _FRAMING_TIPS
        )
        st.markdown(f'<div class="ds-mid">{tips}</div>', unsafe_allow_html=True)
        if st.session_state.get("cam_show_upload"):
            up = st.file_uploader(
                "Choose a picture",
                type=["jpg", "jpeg", "png"],
                key="hw_upload",
                label_visibility="collapsed",
            )
            if up is not None:
                _store_capture(_accept_upload(up))
        with actions_slot():
            first, second = st.columns([3, 2], gap="small")
            with first:
                if st.button(
                    "Take the photo", type="primary", key="pi_capture", use_container_width=True
                ):
                    with st.spinner("Taking the photo…"):
                        shot = _capture_picamera2()
                    if shot is not None:
                        st.session_state["capture_image_bytes"] = shot
                        # Fixed lens, fixed working distance: this is the one
                        # capture path whose millimetres mean anything.
                        st.session_state["capture_from_device"] = True
                        st.rerun()
            with second:
                if st.button("Choose a picture", key="hw_pick", use_container_width=True):
                    st.session_state["cam_show_upload"] = True
                    st.rerun()
        return

    # PC: the capture control is the widget itself, so it lives in the middle
    # of the screen and the action row switches which one is showing.
    source = st.session_state.setdefault("cam_source", "camera")
    with st.container(key="epv-capture"):
        if source == "camera":
            shot = st.camera_input("Camera", key="local_camera", label_visibility="collapsed")
            if shot is not None:
                _store_capture(_sanitize_upload(shot.getvalue()))
        else:
            up = st.file_uploader(
                "Choose a picture",
                type=["jpg", "jpeg", "png"],
                key="cam_upload",
                label_visibility="collapsed",
            )
            if up is not None:
                _store_capture(_accept_upload(up))
            if kind == "mock":
                _render_samples(root)

    with actions_slot():
        first, second = st.columns(2, gap="small")
        with first:
            if st.button(
                "Use the camera",
                type="primary" if source == "camera" else "secondary",
                key="cam_src_cam",
                use_container_width=True,
            ):
                st.session_state["cam_source"] = "camera"
                st.rerun()
        with second:
            if st.button(
                "Choose a picture",
                type="primary" if source == "upload" else "secondary",
                key="cam_src_up",
                use_container_width=True,
            ):
                st.session_state["cam_source"] = "upload"
                st.rerun()


def _render_samples(root: Path) -> None:
    """Demo images, on the mock backend only."""
    samples = list_sample_paths(root)
    if not samples:
        return
    pick = st.selectbox("Or try an example", [label for label, _ in samples], key="mock_sample_pick")
    chosen = dict(samples).get(pick)
    if chosen and st.button("Use this example", key="mock_sample_use", use_container_width=True):
        _store_capture(_sanitize_upload(Path(chosen).read_bytes()))


def render_camera_view(*, root: Path, backend, kind: str) -> None:  # noqa: ARG001 — scan runs in reading_view
    if st.session_state.get("capture_image_bytes"):
        _render_step_two(kind)
    else:
        _render_step_one(root, kind)
