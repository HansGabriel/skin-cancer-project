"""DermaScan v2 — screen router entry."""

from __future__ import annotations

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
from pathlib import Path
from typing import cast

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRONTEND))

from backend.contracts import BackendKind
from logging_config import configure_logging
from navigation import current_route, init_navigation, navigate
from services import settings
from services.auth import (
    STAFF_ROUTES,
    enforce_passcode_gate,
    enforce_staff_gate,
    staff_area_reachable,
)
from services.inference import get_inference_backend
from theme.css import inject_global_css
from theme.tokens import TOKENS
from components.actions import open_actions_slot
from components.instrument import render_disclaimer_strip, render_instrument
from views.assistant_view import render_assistant_view
from views.camera_view import render_camera_view
from views.case_view import render_case_view
from views.folder_view import render_folder_view
from views.history_view import render_history_view
from views.home_view import render_home_view
from views.reading_view import render_reading_view
from views.results_view import render_results_view
from views.settings_view import render_settings_view
from views.staff_view import render_staff_view


def _model_path() -> str:
    return os.environ.get("SKIN_MODEL_PATH", str(ROOT / "models" / "skin_classifier.tflite"))


def _labels_path() -> str:
    return os.environ.get("SKIN_LABELS_PATH", str(ROOT / "models" / "labels.txt"))


def _init_session() -> None:
    """Navigation, then every Settings value — defaults, env and the saved file.

    This used to seed nine keys inline, each of them the ``key`` of a Settings
    widget. Streamlit collects a keyed widget's state on any run where that
    widget is not drawn, so those ``setdefault`` calls were not seeding defaults
    on first use — they were *overwriting the staff's choice* on every screen
    change. services/settings.py explains the whole shape of it; the defaults
    and their justifications now live in its SPEC table.
    """
    init_navigation()
    settings.init_session()


def main() -> None:
    configure_logging()
    st.set_page_config(
        page_title=TOKENS.brand_name,
        page_icon="🧬",
        # The shell paints both bands edge to edge and sizes itself off the
        # viewport, so the centred 1040px column the app used to sit in would
        # only put white gutters either side of it.
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_global_css()
    enforce_passcode_gate()
    _init_session()

    kind = cast(BackendKind, settings.get("inference_backend_kind"))
    backend = get_inference_backend(
        kind, _model_path(), _labels_path(), str(settings.get("pi_base_url_input")).rstrip("/")
    )

    with st.sidebar:
        st.caption("Power user")
        if st.button("Settings", key="side_settings"):
            navigate("settings")
        if st.button("Health check", key="health_btn"):
            health = backend.health()
            log = logging.getLogger("dermascan.app")
            if health.get("status") == "error":
                reason = health.get("reason", "unknown")
                log.error("Health check failed: %s", reason)
                st.error(reason)
            else:
                log.info(
                    "Health check ok backend=%s model=%s",
                    getattr(backend, "backend_id", kind),
                    health.get("model", "?"),
                )
                st.success(f"OK — model: {health.get('model', '?')}")

    route = current_route()

    # The shell: one dark instrument band beside one white page band, both full
    # height. Every screen fills the page band; the instrument is drawn once,
    # here, so no view can forget it or draw a second one.
    # Resolve the gate BEFORE anything is drawn. The instrument band paints the
    # scan the current route is about — for history/folder/case that is a saved
    # participant's photo, and rendering it first put that photo on screen
    # beside the passcode keypad that exists to keep it hidden. `staff_locked`
    # makes the band fall back to its at-rest state.
    staff_locked = route in STAFF_ROUTES and not staff_area_reachable()

    with st.container(key="epv-shell"):
        band, page = st.columns(
            [TOKENS.band_pct, 100 - TOKENS.band_pct], gap="small", vertical_alignment="top"
        )
        with band:
            with st.container(key="epv-band"):
                render_instrument("home" if staff_locked else route, kind=kind)
        with page:
            # The staff readout takes the band dark — that is the design's
            # overlay, achieved by being a route rather than by stacking.
            with st.container(key="epv-page-dark" if route == "staff" else "epv-page"):
                # Three children, and theme/css.py depends on it being exactly
                # three, in this order: the screen (which grows to fill and is
                # the only thing that scrolls), the pinned actions row, and the
                # safety strip. Streamlit wraps every container in an anonymous
                # element, so the only way to pin anything is to make position
                # in this list meaningful.
                #
                # All three are created here, before the screen is dispatched,
                # because DOM order follows creation order — but a container can
                # be filled later. That is what keeps the action buttons out of
                # the scrolling region while still letting each view own them.
                body = st.container(key="epv-body")
                open_actions_slot()
                # Rendered now rather than after dispatch so that a screen which
                # ends the script early — the staff gate calls st.stop() — cannot
                # take the mandated safety line down with it.
                render_disclaimer_strip()
                with body:
                    # Renders its own screen and stops the script when locked.
                    # It runs inside the page band so the keypad appears where
                    # every other screen does, instrument still beside it.
                    enforce_staff_gate(route)
                    _dispatch(route, backend=backend, kind=kind)


def _dispatch(route: str, *, backend, kind: BackendKind) -> None:
    if route == "home":
        render_home_view()
    elif route == "camera":
        render_camera_view(root=ROOT, backend=backend, kind=kind)
    elif route == "reading":
        render_reading_view(backend=backend, kind=kind)
    elif route == "results":
        render_results_view(root=ROOT, model_path=_model_path())
    elif route == "staff":
        render_staff_view(root=ROOT, model_path=_model_path())
    elif route == "history":
        render_history_view()
    elif route == "folder":
        render_folder_view()
    elif route == "case":
        render_case_view()
    elif route == "settings":
        render_settings_view(root=ROOT)
    elif route == "assistant":
        render_assistant_view()
    else:
        navigate("home")


if __name__ == "__main__":
    main()
