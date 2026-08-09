"""Passcode gate for hosted deployments.

If neither ``DERMASCAN_PASSCODE`` env var nor ``st.secrets['dermascan']['passcode']``
is set, the gate is inert — preserves local-dev UX. When a passcode is configured,
``enforce_passcode_gate()`` halts ``app.main()`` until the visitor enters it.
"""

from __future__ import annotations

import hmac
import os
import time

import streamlit as st

from services.kiosk import is_kiosk


def _try_unlock(entered: str | None, expected: str, state_key: str) -> None:
    """Constant-time passcode check; on success flips the session flag and reruns."""
    if entered is None:
        return
    if hmac.compare_digest(entered, expected):
        st.session_state[state_key] = True
        st.session_state[f"{state_key}_at"] = time.monotonic()
        st.rerun()
    else:
        st.error("That code is not right. Try again.")


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
    # Kiosk uses the route-scoped staff gate instead — visitors must be able to
    # scan without a code; only the staff areas are locked there.
    if is_kiosk():
        return
    expected = _configured_passcode()
    if not expected:
        return
    if st.session_state.get("_auth_ok"):
        return

    from theme.tokens import TOKENS

    st.markdown(f"### {TOKENS.brand_name}")
    st.caption("Enter the access code to continue.")
    entered = st.text_input("Access code", type="password", key="_auth_passcode_input")
    if st.button("Unlock", key="_auth_unlock"):
        _try_unlock(entered or "", expected, "_auth_ok")
    st.stop()


# Routes that expose saved body photos or destructive actions. When a passcode is
# configured these are staff-only; Scan/Results stay open so visitors never need a code.
STAFF_ROUTES = ("history", "folder", "case", "settings")


# A kiosk browser session lasts the whole event, so one staff unlock would
# otherwise leave saved photos and the wipe button open to every later visitor.
STAFF_SESSION_SECONDS = float(os.environ.get("DERMASCAN_STAFF_TIMEOUT", "300"))


def _render_staff_unavailable() -> None:
    """Kiosk with no passcode set: refuse the staff area instead of opening it."""
    from navigation import navigate

    st.markdown("### Staff area")
    st.info(
        "This area is closed because no staff code has been set on this device. "
        "Ask the person who set up the kiosk to add one."
    )
    if st.button("← Back to Home", key="_staff_unavailable_back"):
        navigate("home")
    st.stop()


def _staff_unlocked() -> bool:
    """True while a staff unlock is still within its timeout window."""
    since = st.session_state.get("_staff_ok_at")
    if not since:
        return False
    if (time.monotonic() - float(since)) > STAFF_SESSION_SECONDS:
        lock_staff_area()
        return False
    return True


def lock_staff_area() -> None:
    """Drop staff access immediately (timeout, or the Lock button in Settings)."""
    st.session_state.pop("_staff_ok", None)
    st.session_state.pop("_staff_ok_at", None)


def enforce_staff_gate(route: str) -> None:
    if route not in STAFF_ROUTES:
        return
    expected = _configured_passcode()
    if expected is None and is_kiosk():
        # Fail closed: an unattended kiosk with no passcode configured must not
        # expose saved body photos, "Erase all participant data", or Exit.
        _render_staff_unavailable()
        return
    if not expected or _staff_unlocked():
        return

    from navigation import navigate

    st.markdown("### Staff area")
    st.caption("Saved photos and settings are locked. Ask a staff member to open this.")
    if is_kiosk():
        # Keyboard-less kiosk: on-screen keypad instead of a text input.
        from components.keypad import render_numeric_keypad

        _try_unlock(render_numeric_keypad(key="_staff_keypad"), expected, "_staff_ok")
        if st.button("← Back to Home", key="_staff_back"):
            navigate("home")
        st.stop()
    entered = st.text_input("Staff passcode", type="password", key="_staff_passcode_input")
    if st.button("Unlock", key="_staff_unlock"):
        _try_unlock(entered or "", expected, "_staff_ok")
    if st.button("← Back to Home", key="_staff_back_pc"):
        navigate("home")
    st.stop()
