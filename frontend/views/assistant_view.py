"""Offline assistant: doctor-reviewed answers, tap-first, about *this* spot.

Two things changed here from the version that only knew a band string.

**It now carries the scan's measurements.** ``_scan_context`` collects the
label, the ABCDE numbers, and whether the scan was forced, from whichever scan
the assistant is pinned to. ``_current_band`` still exists and still returns the
same pair — the retrieval side is unchanged and its tests pin that — but the
answer can now be led by a sentence built from what was actually measured.

**That sentence is composed in Python, never generated.** It comes from
``services.verdict.signs_sentence``, the same generator the results screen uses
for "what the scan saw", so the chat and the result cannot describe the same
lesion differently. The on-device model may reword approved text; it never sees
the scan and never authors a measurement (``docs/ASSISTANT.md``).

The screen shows one exchange at a time. A growing transcript pushes the
follow-up questions off a 600px panel, and the follow-ups are the whole
interaction on a device with no keyboard.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any

import streamlit as st

from backend.assistant import (
    KnowledgeBase,
    Matcher,
    OllamaRephraser,
    is_lesion_specific,
    load_kb,
    suggested_questions,
)
from backend.contracts import ScanResult
from components.actions import actions_slot
from components.instrument import render_head
from navigation import navigate
from services.format import tone_chip, tone_colors
from services.kiosk import is_kiosk
from services.verdict import resolve_verdict, signs_sentence, verdict_for_saved_scan

_DISCLAIMER = "This is a screening aid, not a diagnosis. Contact a health professional."
_BANDS = ("benign", "pre_cancerous", "malignant")

# Entries whose answer is about the person's own spot get the measured sentence
# in front of them, and they skip the rephraser: its only content guard is a
# timeframe check, so a reworded "17 mm" could come back as "about two
# centimetres" with nothing to catch it. The prefix list itself lives in
# backend.assistant, next to the ordering rule that also depends on it.


@dataclass(frozen=True)
class ScanContext:
    """What the assistant knows about the scan it is answering for."""

    band: str
    label: str | None
    abcde: dict[str, Any] | None
    forced: bool = False
    taken_at: str | None = None

    @property
    def display(self) -> str | None:
        return self.label.upper().replace("_", " ") if self.label else None


@st.cache_resource
def _get_kb_and_matcher(kb_mtime: float) -> tuple[KnowledgeBase, Matcher]:
    kb = load_kb()
    return kb, Matcher(kb)


def _kb_mtime() -> float:
    from backend.assistant import _kb_path

    try:
        return _kb_path().stat().st_mtime
    except OSError:
        return 0.0


def _set_context(ctx: tuple[str, str | None]) -> None:
    """Point the assistant at one scan, clearing the chat when it changes.

    The chat has to reset on a context switch. Without this, asking about a
    saved scan from March and then opening the assistant on a fresh result
    leaves March's questions and answers sitting above the new verdict, which
    reads as though they were said about the new one.
    """
    if st.session_state.get("assistant_context") != ctx:
        st.session_state["assistant_chat"] = []
    st.session_state["assistant_context"] = ctx


def ask_about_last_result() -> None:
    """Open the assistant on the scan currently on screen."""
    st.session_state.pop("assistant_saved_scan", None)
    _set_context(("live", None))
    navigate("assistant")


def ask_about_saved_scan(scan) -> None:
    """Open the assistant on a stored scan (``services.storage.Scan``).

    The scan row is held rather than its case id: everything the assistant
    needs is on the row, and keeping it means the banner can name the date the
    participant is actually asking about.
    """
    st.session_state["assistant_saved_scan"] = scan
    _set_context(("saved", str(scan.id)))
    navigate("assistant")


def clear_scan_context() -> None:
    """Drop any saved-scan pin — used by the nav key, which means "current".

    Without this the key is a trap: pin a saved scan, run a new check, tap
    Questions, and the assistant still answers for the old one.
    """
    st.session_state.pop("assistant_saved_scan", None)
    _set_context(("live", None))


def _scan_context() -> ScanContext:
    """Everything the assistant is allowed to know about the pinned scan.

    A pinned saved scan wins over ``last_result``: arriving from the case
    screen must answer for the scan that was tapped, not for whatever happened
    to be scanned last in this session.
    """
    saved = st.session_state.get("assistant_saved_scan")
    if saved is not None:
        label = str(getattr(saved, "label", ""))
        abcde = None
        raw = getattr(saved, "abcd_json", None)
        if raw:
            try:
                abcde = json.loads(raw)
            except (TypeError, ValueError):
                abcde = None
        e_raw = getattr(saved, "e_json", None)
        if abcde is not None and e_raw:
            try:
                abcde["E"] = json.loads(e_raw)
            except (TypeError, ValueError):
                pass
        if label in _BANDS:
            return ScanContext(label, label, abcde, taken_at=str(getattr(saved, "taken_at", ""))[:10])
        return ScanContext("general", None, abcde)

    pl = st.session_state.get("last_result")
    if not isinstance(pl, dict):
        return ScanContext("general", None, None)
    sr = pl.get("scan_result")
    abcde = pl.get("abcde")
    forced = bool(pl.get("forced"))
    if isinstance(sr, ScanResult) and sr.label in _BANDS:
        return ScanContext(sr.label, sr.label, abcde, forced=forced)
    return ScanContext("general", None, abcde, forced=forced)


def _current_band() -> tuple[str, str | None]:
    """(kb band, display label). Kept as the retrieval-side interface."""
    ctx = _scan_context()
    return ctx.band, ctx.display


def _answer(
    kb: KnowledgeBase, matcher: Matcher, query: str, ctx: ScanContext
) -> tuple[str, bool, bool]:
    """(text, append_disclaimer, ai_reworded) for a user query."""
    entry, _score = matcher.match(query, ctx.band)
    if entry is None:
        return kb.fallback_answer, True, False

    lesion_specific = is_lesion_specific(entry.id)
    text, reworded = entry.answer, False
    if lesion_specific:
        measured = signs_sentence(ctx.abcde)
        if measured:
            text = f"{measured}\n\n{text}"
        # Deliberately not reworded — see the comment above.
        return text, entry.always_append_disclaimer, False

    if st.session_state.get("assistant_gemma_enabled"):
        text, reworded = OllamaRephraser().reword(entry.answer)
    return text, entry.always_append_disclaimer, reworded


def _render_answer_card(msg: dict) -> None:
    badge_ink = "#0F7A6C"
    st.markdown(
        '<div style="margin-top:12px;padding:16px 18px;border-radius:12px;background:#F7F9FC">'
        '<div class="ds-note-label" style="color:#5B6675">You asked</div>'
        f'<div style="margin-top:5px;font-weight:600">{msg["question"]}</div>'
        f'<div style="margin-top:10px;line-height:1.5">{msg["content"]}</div>'
        f'<div style="display:flex;align-items:center;gap:7px;margin-top:10px">'
        f'<span style="width:7px;height:7px;border-radius:999px;background:{badge_ink}"></span>'
        f'<span style="font-weight:600;letter-spacing:.12em;color:{badge_ink};font-size:12px">'
        f'{msg["badge"]}</span></div></div>',
        unsafe_allow_html=True,
    )


def render_assistant_view() -> None:
    ctx = _scan_context()
    saved = st.session_state.get("assistant_saved_scan")

    chip = ""
    if ctx.display:
        v = (
            verdict_for_saved_scan(saved)
            if saved is not None
            else (st.session_state.get("last_result") or {}).get("verdict")
        )
        if v is None:
            pl = st.session_state.get("last_result") or {}
            v = resolve_verdict(pl.get("scan_result"), pl.get("quality"))
        ink, fill = tone_colors(v.tone)
        chip = f'<span class="ds-pill" style="background:{fill};color:{ink}">{tone_chip(v.tone)}</span>'

    render_head("Questions · reviewed answers", "Ask about your result", extra=chip)
    if saved is not None and ctx.taken_at:
        st.caption(f"Asking about a saved scan from {ctx.taken_at}.")
    elif ctx.display:
        st.caption("Asking about your last result.")
    else:
        st.caption("General questions — run a check to ask about a specific spot.")

    try:
        kb, matcher = _get_kb_and_matcher(_kb_mtime())
    except (OSError, ValueError) as exc:
        st.error(f"Assistant knowledge base unavailable: {exc}")
        return
    if not kb.entries:
        st.info(
            "The assistant's answers are being reviewed by a health professional "
            "and will appear here soon. In the meantime, please direct any "
            "questions about your result to a health worker."
        )
        return

    chat: list[dict] = st.session_state.setdefault("assistant_chat", [])

    def _ask(question: str, entry_id: str | None = None) -> None:
        text, disclaim, reworded = _answer(kb, matcher, question, ctx)
        if disclaim:
            # The card is rendered as raw HTML, so markdown emphasis would
            # show as literal asterisks — 19 of the 22 entries set
            # always_append_disclaimer.
            text = f"{text}\n\n<em>{_DISCLAIMER}</em>"
        badge = (
            "Friendly wording by on-device AI" if reworded else "Reviewed by a health professional"
        )
        chat.append(
            {
                # Escaped: this lands in unsafe_allow_html markup, and on a PC
                # the question is free text the visitor typed.
                "question": html.escape(question),
                "content": text.replace("\n\n", "<br><br>"),
                "badge": badge,
                "entry_id": entry_id,
            }
        )
        st.rerun()

    # Only the most recent exchange. Earlier ones stay in session state so the
    # suggestion list keeps excluding what was already asked.
    if chat:
        _render_answer_card(chat[-1])

    asked_ids = {m.get("entry_id") for m in chat if m.get("entry_id")}
    st.markdown(
        '<p class="ds-section-title" style="margin-top:18px">'
        f'{"Or tap another question" if chat else "Tap a question"}</p>',
        unsafe_allow_html=True,
    )
    for entry_id, question in suggested_questions(kb, ctx.band, exclude_ids=asked_ids):
        if st.button(question, key=f"assist_sq_{entry_id}", use_container_width=True):
            _ask(question, entry_id)

    # Free-text input only where a keyboard exists (PC / Streamlit Cloud).
    if not is_kiosk():
        typed = st.chat_input("Ask about your result…")
        if typed:
            _ask(typed.strip())

    with actions_slot():
        back, clear = st.columns([3, 2], gap="small")
        with back:
            has_result = isinstance(st.session_state.get("last_result"), dict)
            if st.button(
                "Back to the result" if has_result else "Back to start",
                type="primary",
                key="assist_back",
                use_container_width=True,
            ):
                navigate("results" if has_result else "home")
        with clear:
            if chat and st.button("Start over", key="assist_clear", use_container_width=True):
                st.session_state["assistant_chat"] = []
                st.rerun()
