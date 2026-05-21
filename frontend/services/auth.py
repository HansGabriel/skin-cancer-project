"""Passcode gate for hosted deployments.

If neither ``DERMASCAN_PASSCODE`` env var nor ``st.secrets['dermascan']['passcode']``
is set, the gate is inert — preserves local-dev UX. When a passcode is configured,
``enforce_passcode_gate()`` halts ``app.main()`` until the visitor enters it.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st


def _configured_passcode() -> str | None:
    env = os.environ.get("DERMASCAN_PASSCODE")
    if env:
        return env
    try:
        secret = st.secrets["dermascan"]["passcode"]
    except (KeyError, FileNotFoundError, AttributeError):
        return None
    return str(secret) if secret else None


def enforce_passcode_gate() -> None:
    expected = _configured_passcode()
    if not expected:
        return
    if st.session_state.get("_auth_ok"):
        return

    st.markdown("### DermaScan AI")
    st.caption("Enter the access passcode to continue.")
    entered = st.text_input("Passcode", type="password", key="_auth_passcode_input")
    if st.button("Unlock", key="_auth_unlock"):
        if hmac.compare_digest(entered or "", expected):
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect passcode.")
    st.stop()
