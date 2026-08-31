from __future__ import annotations

from pathlib import Path

import streamlit as st

from backend import pi_backend_enabled
from components.actions import actions_slot
from components.instrument import render_head
from navigation import navigate
from services import settings
from services.kiosk import is_kiosk, request_quit
from services.storage import data_dir, get_storage
from theme.tokens import TOKENS


def _toggle(name: str, label: str, *, help: str | None = None) -> None:
    """A toggle bound to a setting rather than owning its own state.

    Every widget on this screen follows the same three-part shape and it is not
    optional: the Streamlit ``key`` is ``ui_<name>``, the initial value comes
    from the canonical key, and ``on_change`` commits back into it. Using the
    canonical key as the widget key — which is what this screen used to do — is
    the defect services/settings.py exists to fix.
    """
    st.toggle(
        label,
        value=bool(settings.get(name)),
        key=settings.widget_key(name),
        on_change=settings.commit,
        args=(name,),
        help=help,
    )


def render_settings_view(*, root: Path) -> None:  # noqa: ARG001 — kept for the router's uniform signature
    render_head("Settings · staff", "Settings")
    store = get_storage()

    # ── Plain settings (everything an ordinary user / health worker needs) ──
    # One switch, not two. It used to be labelled "Refuse photos that are not
    # very clear" and it only ever touched the *capture* gate (blur and light) —
    # so staff who turned it on expecting the scanner to stop reading things
    # that are not lesions got no such thing. It now tightens both gates, and
    # the label says which is which.
    #
    # Off is the default, and off is not "no checking": services.lesion_gate
    # runs unconditionally and refuses bare skin, screens and scenes on its own.
    # This only raises the bar, and raising it costs real lesions — blocking
    # every slightly-soft photo was measured refusing 72% of genuine HAM10000
    # images.
    _toggle(
        "strict_checking",
        "Check photos strictly",
        help=(
            "Off: a slightly blurry or dim photo is still read, and photos that "
            "are plainly not a spot on skin are still refused. On: borderline "
            "photos are refused too. Turning this on will reject some real "
            "lesions."
        ),
    )

    # Assistant wording (on-device AI). Answers stay reviewed either way.
    # MVP 2 locked design: canned doctor-reviewed text by default — the LLM
    # reword layer is opt-in per session, never on by default.
    from backend.assistant import OllamaRephraser

    _toggle(
        "assistant_gemma_enabled",
        "Friendlier assistant wording (on-device AI)",
        help=(
            "Rewords the assistant's reviewed answers in a friendlier tone. The "
            "medical content never changes, and answers that quote this spot's "
            "own measurements are never reworded."
        ),
    )
    if settings.get("assistant_gemma_enabled"):
        h = OllamaRephraser().health()
        if h["status"] == "ok":
            st.caption("Ready")
        else:
            st.caption("Not available on this device — answers are shown as written")

    mb = store.storage_size_bytes() / (1024 * 1024)
    st.markdown(
        f'<div class="ds-row"><span class="ds-row-grow">Saved scans on this device</span>'
        f'<span style="font-weight:600">{mb:.1f} MB</span></div>',
        unsafe_allow_html=True,
    )

    # ── Developer / advanced — collapsed so it never confuses end users ──
    with st.expander("Advanced (developer)"):
        pi_enabled = pi_backend_enabled()
        backend_options = ("local", "mock", "pi") if pi_enabled else ("local", "mock")
        if not pi_enabled and settings.get("inference_backend_kind") == "pi":
            # Both keys, deliberately. A live widget's own state wins over the
            # `index=` below within the same run, so writing only the canonical
            # key would leave the radio still showing "pi" until the next rerun.
            settings.set_value("inference_backend_kind", "local")
            st.session_state[settings.widget_key("inference_backend_kind")] = "local"
        backend_labels = {
            "local": "PC — browser camera or upload (recommended)",
            "mock": "PC — samples + upload (same model, no camera label)",
            "pi": "Raspberry Pi — Pi camera over LAN",
        }
        st.radio(
            "Backend",
            backend_options,
            index=backend_options.index(str(settings.get("inference_backend_kind"))),
            format_func=lambda k: backend_labels[k],
            key=settings.widget_key("inference_backend_kind"),
            on_change=settings.commit,
            args=("inference_backend_kind",),
            help="Camera screen always offers webcam + upload on PC. Pi uses the device on the network.",
        )
        if pi_enabled:
            st.text_input(
                "Pi base URL",
                value=str(settings.get("pi_base_url_input")),
                key=settings.widget_key("pi_base_url_input"),
                on_change=settings.commit,
                args=("pi_base_url_input",),
            )
        st.text_input(
            "SKIN_KERAS_PATH",
            value=str(settings.get("SKIN_KERAS_PATH_UI")),
            key=settings.widget_key("SKIN_KERAS_PATH_UI"),
            on_change=settings.commit,
            args=("SKIN_KERAS_PATH_UI",),
        )
        st.number_input(
            "pixels_per_mm",
            0.1,
            100.0,
            value=float(settings.get("pixels_per_mm_ui")),
            step=0.5,
            key=settings.widget_key("pixels_per_mm_ui"),
            on_change=settings.commit,
            args=("pixels_per_mm_ui",),
        )
        _toggle(
            "preprocess_enabled",
            "Enhance image for ABCDE only (color + hair removal)",
            help="Improves asymmetry/border measurements. The CNN always receives the original photo.",
        )
        _toggle(
            "preprocess_debug",
            "Debug: show original vs ABCDE-enhanced on the staff readout",
        )
        kind = settings.get("inference_backend_kind")
        # No os.environ write here. SKIN_TTA is pushed by settings.apply_env()
        # on every run instead: doing it from this screen meant the environment
        # and the toggle disagreed the moment staff navigated away.
        _toggle(
            "tta_toggle",
            "Test-time augmentation (slower)",
            help=(
                "Four views averaged. docs/METRICS.md measured the deployed 0.911 "
                "cancer sensitivity with this ON; turning it off serves a "
                "configuration nobody has measured."
            ),
        )
        tta = bool(settings.get("tta_toggle"))
        if kind == "pi" and tta:
            st.caption("Pi backend: TTA costs about 0.6 s per scan on the device.")
        if st.button("Clear documents cache"):
            store.clear_cache()
            st.success("Cache cleared.")
        st.caption(f"Data: `{data_dir()}` · Env: DERMASCAN_DATA_DIR, SKIN_MODEL_PATH")
        if not is_kiosk() and st.text_input("Type DELETE to reset") == "DELETE":
            if st.button("Reset all data"):
                store.reset_all()
                st.session_state.pop("last_result", None)
                st.success("Reset complete.")
                st.rerun()

    if is_kiosk():
        # No free-text confirm on the keyboard-less kiosk — two-tap reset.
        wipe_armed = st.checkbox(
            "End event — I want to erase all saved scans", key="reset_confirm"
        )
    else:
        wipe_armed = False

    with actions_slot():
        lock, wipe = st.columns(2, gap="small")
        with lock:
            # Lock the staff area back up without waiting for the timeout — the
            # kiosk holds one browser session all event, so leaving it open
            # hands saved photos and the wipe button to whoever walks up next.
            #
            # Not an if/elif pair: reaching Settings at all means the staff gate
            # is open, so `_staff_ok` is effectively always true here and an
            # `elif` made the kiosk Exit button dead in every configuration.
            #
            # This is now the ONLY way out of the kiosk: the nav key was removed
            # (see components/bottom_nav.py). One tap rather than two is fine
            # here — the staff passcode is already the guard against a stray
            # touch, which is what the old two-tap confirm was standing in for.
            if st.session_state.get("_staff_ok") and st.button(
                "Lock the staff area", key="staff_lock", use_container_width=True
            ):
                from services.auth import lock_staff_area

                lock_staff_area()
                navigate("home")
            if is_kiosk() and st.button(
                f"Exit {TOKENS.brand_name}", key="kiosk_exit", use_container_width=True
            ):
                request_quit()
                st.stop()
        with wipe:
            if is_kiosk() and st.button(
                "End event — erase all",
                key="kiosk_wipe",
                disabled=not wipe_armed,
                use_container_width=True,
            ):
                store.reset_all()
                st.session_state.pop("last_result", None)
                st.success("All saved scans erased.")
                st.rerun()
