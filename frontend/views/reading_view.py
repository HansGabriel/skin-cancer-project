"""Step 3: the scan runs, and you can watch it run.

This used to be a ``st.spinner("Checking the spot…")`` on the capture screen —
one opaque wait that could last thirty seconds on a Pi with no indication that
anything was happening, or that the device had not simply frozen.

**How it works.** Streamlit sends each element to the browser as it is
produced, so everything rendered *before* a blocking call is already on screen
while that call runs. This view therefore draws the whole reading screen, then
runs the pipeline, then navigates. The three checklist rows are driven by the
pipeline's real stage callback (``services.pipeline.Stages.on_enter``), so a
tick means the work is genuinely done.

**There is no Cancel button.** Streamlit cannot interrupt a running script from
a widget — the click is not processed until the script finishes, by which point
the scan has already completed. A button that does nothing for thirty seconds
and then silently does nothing is worse on a medical device than no button, so
the design's Cancel is deliberately absent.
"""

from __future__ import annotations

import streamlit as st

from components.instrument import render_head
from navigation import navigate
from services.scan_flow import run_scan_and_store

# Pipeline stage → which of the three visible steps it belongs to. The names
# come from services.pipeline._stage; anything not listed (decode) is folded
# into the first step, which is already showing by the time it runs.
_STEP_OF_STAGE = {
    "quality": 0,
    "skin": 0,
    "enhance": 1,
    "segment": 1,
    "spot": 1,
    "model": 2,
    "abcde": 2,
}

_STEP_TEXT = (
    "Checked the photo and found skin",
    "Found the spot and drew its outline",
    "Compared it with the on-device model",
)


def _step_row(index: int, current: int) -> str:
    if index < current:
        state, dot = "is-done", ""
    elif index == current:
        state, dot = "is-now", ""
    else:
        state, dot = "", ""
    return (
        f'<div class="ds-step {state}"><span class="ds-step-dot"></span>'
        f"<span>{_STEP_TEXT[index]}</span>{dot}</div>"
    )


def _render_checklist(slot, current: int) -> None:
    slot.markdown(
        "".join(_step_row(i, current) for i in range(len(_STEP_TEXT))),
        unsafe_allow_html=True,
    )


def render_reading_view(*, backend, kind: str) -> None:  # noqa: ARG001 — kind is the backend's business
    image_bytes = st.session_state.get("capture_image_bytes")
    forced = bool(st.session_state.pop("force_rescan", False))

    # Arriving here with nothing to scan means a stale link or a reload after
    # the photo was dropped. The Pi backend is the exception: it captures for
    # itself, so "no bytes" is its normal case.
    if image_bytes is None and kind != "pi":
        navigate("camera")
        return

    render_head(
        "Step 3 of 3 · reading",
        "Reading the spot…",
        "Everything happens on this device. No internet needed.",
    )
    st.markdown(
        '<div class="ds-progress"><div class="ds-progress-fill" style="width:100%"></div></div>',
        unsafe_allow_html=True,
    )
    checklist = st.empty()
    _render_checklist(checklist, 0)

    # Streamlit has flushed everything above by now; the scan blocks below it.
    pixels = float(st.session_state.get("pixels_per_mm_ui", 10.0))
    keras = str(st.session_state.get("SKIN_KERAS_PATH_UI", ""))
    strict_q = bool(st.session_state.get("strict_quality_gate", False))
    case_id = st.session_state.get("pending_case_id")
    seen = {"step": 0}

    def _on_stage(name: str) -> None:
        step = _STEP_OF_STAGE.get(name)
        if step is None or step <= seen["step"]:
            return
        seen["step"] = step
        _render_checklist(checklist, step)

    pl = run_scan_and_store(
        backend,
        image_bytes,
        pixels_per_mm=pixels,
        strict_quality=strict_q,
        keras_path=keras,
        case_id=str(case_id) if case_id else None,
        force=forced,
        on_stage=_on_stage,
    )

    # Keep the photo when the scan was refused: the result screen offers "Check
    # it anyway", and that retry needs the original bytes. Discard it otherwise,
    # so a finished scan leaves no skin photo in session state (docs/PRIVACY.md).
    if not pl.get("blocked"):
        st.session_state.pop("capture_image_bytes", None)
    if forced:
        pl["forced"] = True
    st.session_state["last_result"] = pl
    navigate("results")
