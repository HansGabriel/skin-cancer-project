"""On-screen numeric keypad for the keyboard-less kiosk (SKIN_KIOSK=1).

Streamlit-native (st.button grid + session-state buffer) — no system OSK
dependency. Used by the passcode gate; reusable for any numeric entry.
"""

from __future__ import annotations

import streamlit as st

from theme.tokens import TOKENS as T

_ROWS = (("1", "2", "3"), ("4", "5", "6"), ("7", "8", "9"), ("⌫", "0", "✓"))


def render_numeric_keypad(*, key: str, max_len: int = 12) -> str | None:
    """Render a 3x4 keypad. Returns the buffered value when ✓ is tapped, else None.

    The in-progress value lives in ``st.session_state[f"{key}_buf"]`` and is
    shown masked above the keys.
    """
    buf_key = f"{key}_buf"
    buf = st.session_state.get(buf_key, "")

    st.markdown(
        f'<div style="text-align:center;font-size:{T.font_xl}px;letter-spacing:8px;'
        f'min-height:40px;padding:4px 0">{"●" * len(buf) or "&nbsp;"}</div>',
        unsafe_allow_html=True,
    )

    submitted: str | None = None
    for r, row in enumerate(_ROWS):
        cols = st.columns(3)
        for c, label in enumerate(row):
            if not cols[c].button(label, key=f"{key}_{r}{c}", use_container_width=True):
                continue
            if label == "⌫":
                st.session_state[buf_key] = buf[:-1]
                st.rerun()
            elif label == "✓":
                submitted = buf
                st.session_state[buf_key] = ""
            elif len(buf) < max_len:
                st.session_state[buf_key] = buf + label
                st.rerun()
    return submitted
