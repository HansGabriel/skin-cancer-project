"""The assistant must answer for the scan the user actually tapped.

Two entry points pin a specific scan — "Ask a question about this" on the
results screen and "Ask about this" on each saved scan in a case — while the
"Questions" nav tab means "whatever I have now". Getting that wrong is not a
cosmetic bug: a case can hold several scans with different verdicts, so
answering a MALIGNANT question set for a scan the user believes is BENIGN is
the exact false-reassurance this app is built to avoid.

These tests drive the band resolution directly rather than through Streamlit,
because the failure is in which scan gets picked, not in how it renders.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import streamlit as st

from views.assistant_view import (
    _current_band,
    ask_about_saved_scan,
    clear_scan_context,
)


@dataclass
class _Scan:
    """Just the fields the assistant reads off ``services.storage.Scan``."""

    id: str = "scan-1"
    label: str = "malignant"
    confidence: float = 0.83
    taken_at: str = "2026-03-04T10:00:00"


@pytest.fixture(autouse=True)
def _clean_session():
    for key in ("assistant_saved_scan", "assistant_context", "assistant_chat", "last_result"):
        st.session_state.pop(key, None)
    yield
    for key in ("assistant_saved_scan", "assistant_context", "assistant_chat", "last_result"):
        st.session_state.pop(key, None)


def test_pinned_saved_scan_sets_its_own_band() -> None:
    st.session_state["assistant_saved_scan"] = _Scan(label="benign")
    assert _current_band() == ("benign", "BENIGN")


def test_pinned_saved_scan_beats_a_newer_live_result() -> None:
    """The regression this pinning exists to prevent.

    A participant opens a saved BENIGN scan and asks about it. If the live
    result from a later, unrelated scan won, they would be reading malignant
    answers under a benign photo.
    """
    st.session_state["last_result"] = {"scan_result": _Scan(label="malignant")}
    st.session_state["assistant_saved_scan"] = _Scan(label="benign")
    band, _label = _current_band()
    assert band == "benign", "the pinned saved scan must win over last_result"


def test_unknown_stored_label_falls_back_to_general() -> None:
    """A label the KB has no band for must not raise or guess."""
    st.session_state["assistant_saved_scan"] = _Scan(label="something_new")
    assert _current_band() == ("general", None)


def test_nav_tab_unpins_a_saved_scan() -> None:
    """The tab means "current" — leaving a pin set makes it answer stale."""
    st.session_state["assistant_saved_scan"] = _Scan(label="malignant")
    clear_scan_context()
    assert "assistant_saved_scan" not in st.session_state
    assert _current_band() == ("general", None)


def test_switching_scan_clears_the_conversation(monkeypatch) -> None:
    """Old questions must not sit above a new verdict.

    Without this the chat reads as though March's answers were said about the
    scan now on screen.
    """
    monkeypatch.setattr("views.assistant_view.navigate", lambda *_a, **_k: None)
    st.session_state["assistant_chat"] = [{"role": "user", "content": "is it serious?"}]

    ask_about_saved_scan(_Scan(id="scan-A", label="benign"))
    assert st.session_state["assistant_chat"] == [], "chat survived a context switch"

    st.session_state["assistant_chat"] = [{"role": "user", "content": "what now?"}]
    ask_about_saved_scan(_Scan(id="scan-A", label="benign"))
    assert st.session_state["assistant_chat"], "re-opening the SAME scan wiped the chat"

    ask_about_saved_scan(_Scan(id="scan-B", label="malignant"))
    assert st.session_state["assistant_chat"] == [], "chat leaked from scan-A into scan-B"
