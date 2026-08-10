from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from backend import pi_backend_enabled
from components.mobile_frame import mobile_frame
from components.primary_button import render_back_link
from navigation import navigate
from services.kiosk import is_kiosk, request_quit
from services.storage import data_dir, get_storage
from theme.tokens import TOKENS


def render_settings_view(*, root: Path) -> None:
    with mobile_frame():
        if render_back_link("Home", key="set_back"):
            navigate("home")
        st.markdown("### Settings")
        store = get_storage()

        # ── Plain settings (everything an ordinary user / health worker needs) ──
        # Off by default. services.lesion_gate is what stops junk input now, and
        # blocking every slightly-soft photo was rejecting real lesions.
        st.toggle(
            "Only accept very clear photos",
            key="strict_quality_gate",
            help=(
                "Photos that are slightly blurry or dim are still checked, with a "
                "note. Turn this on to reject them instead."
            ),
        )

        # Assistant wording (on-device AI). Answers stay reviewed either way.
        # MVP 2 locked design: canned doctor-reviewed text by default — the LLM
        # reword layer is opt-in per session, never on by default.
        from backend.assistant import OLLAMA_MODEL, OllamaRephraser

        st.session_state.setdefault("assistant_gemma_enabled", False)
        st.toggle(
            "Friendlier assistant wording (on-device AI)",
            key="assistant_gemma_enabled",
            help="Rewords the assistant's reviewed answers in a friendlier tone. The medical content never changes.",
        )
        if st.session_state["assistant_gemma_enabled"]:
            h = OllamaRephraser().health()
            if h["status"] == "ok":
                st.caption("🟢 Ready")
            else:
                st.caption("⚪ Not available on this device — answers are shown as written")

        mb = store.storage_size_bytes() / (1024 * 1024)
        st.caption(f"Saved scans use {mb:.1f} MB of storage.")

        st.divider()
        # Lock the staff area back up without waiting for the timeout — the
        # kiosk holds one browser session all event, so leaving it open hands
        # saved photos and the wipe button to whoever walks up next.
        if st.session_state.get("_staff_ok"):
            if st.button("Lock the staff area", key="staff_lock"):
                from services.auth import lock_staff_area

                lock_staff_area()
                navigate("home")

        if is_kiosk():
            # Kiosk: Exit lives here (reachable from every screen via bottom nav).
            # No free-text confirm on the keyboard-less kiosk — two-tap reset instead.
            if st.button(f"Exit {TOKENS.brand_name}", key="kiosk_exit"):
                request_quit()
                st.stop()
            if st.checkbox(
                "End event — I want to erase all saved scans", key="reset_confirm"
            ) and st.button("Erase all participant data"):
                store.reset_all()
                st.session_state.pop("last_result", None)
                st.success("All saved scans erased.")
                st.rerun()
        elif st.text_input("Type DELETE to reset") == "DELETE" and st.button("Reset all data"):
            store.reset_all()
            st.session_state.pop("last_result", None)
            st.success("Reset complete.")
            st.rerun()

        # ── Developer / advanced — collapsed so it never confuses end users ──
        with st.expander("🔧 Advanced (developer)"):
            pi_enabled = pi_backend_enabled()
            backend_options = ("local", "mock", "pi") if pi_enabled else ("local", "mock")
            if not pi_enabled and st.session_state.get("inference_backend_kind") == "pi":
                st.session_state["inference_backend_kind"] = "local"
            backend_labels = {
                "local": "PC — browser camera or upload (recommended)",
                "mock": "PC — samples + upload (same model, no camera label)",
                "pi": "Raspberry Pi — Pi camera over LAN",
            }
            st.radio(
                "Backend",
                backend_options,
                format_func=lambda k: backend_labels[k],
                key="inference_backend_kind",
                help="Camera screen always offers webcam + upload on PC. Pi uses the device on the network.",
            )
            if pi_enabled:
                st.text_input("Pi base URL", key="pi_base_url_input")
            if "SKIN_KERAS_PATH_UI" not in st.session_state:
                st.session_state["SKIN_KERAS_PATH_UI"] = os.environ.get(
                    "SKIN_KERAS_PATH", str(root / "models" / "skin_classifier_full.keras")
                )
            st.text_input("SKIN_KERAS_PATH", key="SKIN_KERAS_PATH_UI")
            if "pixels_per_mm_ui" not in st.session_state:
                st.session_state["pixels_per_mm_ui"] = 10.0
            st.number_input("pixels_per_mm", 0.1, 100.0, step=0.5, key="pixels_per_mm_ui")
            st.toggle(
                "Enhance image for ABCDE only (color + hair removal)",
                key="preprocess_enabled",
                help="Improves asymmetry/border measurements. The CNN always receives the original photo.",
            )
            st.toggle(
                "Debug: show original vs ABCDE-enhanced on Results",
                key="preprocess_debug",
            )
            kind = st.session_state.get("inference_backend_kind", "mock")
            tta_default = kind != "pi"
            if "tta_toggle" not in st.session_state:
                st.session_state["tta_toggle"] = tta_default
            tta = st.toggle(
                "Test-time augmentation (slower)",
                key="tta_toggle",
                help="Recommended off for Raspberry Pi (latency).",
            )
            os.environ["SKIN_TTA"] = "1" if tta else "0"
            if kind == "pi" and tta:
                st.caption("Pi backend: TTA increases latency on the device.")
            if st.button("Clear documents cache"):
                store.clear_cache()
                st.success("Cache cleared.")
            st.caption(f"Data: `{data_dir()}` · Env: DERMASCAN_DATA_DIR, SKIN_MODEL_PATH")
