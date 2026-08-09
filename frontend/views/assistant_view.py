"""Offline assistant view: doctor-reviewed answers, tap-first UI, optional Gemma reword."""

from __future__ import annotations

import streamlit as st

from backend.assistant import KnowledgeBase, Matcher, OllamaRephraser, load_kb, suggested_questions
from backend.contracts import ScanResult
from components.app_bar import render_disclaimer_footer
from components.mobile_frame import mobile_frame
from components.urgency_pill import render_verdict_pill
from services.kiosk import is_kiosk
from services.verdict import resolve_verdict

_DISCLAIMER = "This is a screening aid, not a diagnosis. Contact a health professional."


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


def _current_band() -> tuple[str, str | None]:
    """(kb band, display label) from the last scan, else general."""
    pl = st.session_state.get("last_result")
    sr = pl.get("scan_result") if isinstance(pl, dict) else None
    if isinstance(sr, ScanResult) and sr.label in ("benign", "pre_cancerous", "malignant"):
        return sr.label, sr.label.upper().replace("_", " ")
    return "general", None


def _answer(kb: KnowledgeBase, matcher: Matcher, query: str, band: str) -> tuple[str, bool, bool]:
    """(text, append_disclaimer, ai_reworded) for a user query."""
    entry, _score = matcher.match(query, band)
    if entry is None:
        return kb.fallback_answer, True, False
    text, reworded = entry.answer, False
    if st.session_state.get("assistant_gemma_enabled"):
        text, reworded = OllamaRephraser().reword(entry.answer)
    return text, entry.always_append_disclaimer, reworded


def render_assistant_view() -> None:
    with mobile_frame():
        st.markdown("### 💬 Assistant")
        band, band_label = _current_band()
        if band_label:
            # Same verdict object the results screen rendered — never a second
            # wording derived from the composite risk band.
            pl = st.session_state.get("last_result") or {}
            v = pl.get("verdict") or resolve_verdict(pl.get("scan_result"), pl.get("quality"))
            render_verdict_pill(v)
            st.caption("Asking about your last result.")
        else:
            st.caption("General questions — run a scan to ask about a specific result.")

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
        for msg in chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("badge"):
                    st.caption(msg["badge"])

        def _ask(question: str) -> None:
            text, disclaim, reworded = _answer(kb, matcher, question, band)
            if disclaim:
                text = f"{text}\n\n*{_DISCLAIMER}*"
            badge = "✨ Friendly wording by on-device AI" if reworded else "✓ Reviewed answer"
            chat.append({"role": "user", "content": question})
            chat.append({"role": "assistant", "content": text, "badge": badge})
            st.rerun()

        # Primary input: tappable suggested questions (kiosk has no keyboard).
        asked_ids = {m.get("entry_id") for m in chat if m.get("entry_id")}
        st.markdown("**Tap a question:**")
        for entry_id, question in suggested_questions(kb, band, exclude_ids=asked_ids):
            if st.button(question, key=f"assist_sq_{entry_id}", use_container_width=True):
                _ask(question)

        if chat and st.button("Clear conversation", key="assist_clear"):
            st.session_state["assistant_chat"] = []
            st.rerun()

        # Free-text input only where a keyboard exists (PC / Streamlit Cloud).
        if not is_kiosk():
            typed = st.chat_input("Ask about your result…")
            if typed:
                _ask(typed.strip())

        render_disclaimer_footer()
