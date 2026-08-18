"""The assistant answers about *this* spot, using numbers it did not invent.

Before this, the whole scanner-to-chat interface was one string: the label. A
person could ask "why was my spot flagged?" seconds after the screen showed
them A 0.37 / B 4.58 / C 2 / D 17.41 mm, and get a paragraph that could have
been written before the device existed.

The safety property that must not be lost while fixing that: the measured
sentence is **composed in Python from the measurements**, never generated. The
on-device model (Gemma 3 270M via Ollama) may reword approved text; it never
sees scan data and never authors a number — see docs/ASSISTANT.md.
"""

from __future__ import annotations

import json
import os

import pytest

from backend.assistant import (
    LESION_SPECIFIC_PREFIXES,
    is_lesion_specific,
    load_kb,
    suggested_questions,
)
from services.verdict import signs_sentence


@pytest.fixture(scope="module")
def kb():
    # The shipped entries have no reviewed_by/reviewed_date yet, so the
    # doctor-review gate hides them unless dev mode is on.
    os.environ["SKIN_KB_DEV"] = "1"
    try:
        yield load_kb()
    finally:
        os.environ.pop("SKIN_KB_DEV", None)


ABCDE = {
    "A": {"value": 0.37, "tier": 2, "verdict": "suspicious"},
    "B": {"value": 4.58, "tier": 2, "verdict": "suspicious"},
    "C": {"value": 2, "tier": 1, "verdict": "borderline"},
    "D": {"value": 17.41, "tier": 2, "verdict": "suspicious"},
    "E": {"value": None, "tier": 0, "verdict": "needs history"},
}


def test_lesion_specific_entries_exist_and_are_marked(kb) -> None:
    marked = [e.id for e in kb.entries if is_lesion_specific(e.id)]
    assert marked, "no entry answers about the person's own spot"
    for entry_id in marked:
        assert entry_id.startswith(LESION_SPECIFIC_PREFIXES)


def test_they_are_offered_before_other_general_questions(kb) -> None:
    """A four-button budget was being spent on "What is melanoma?" while
    "Why was my spot flagged?" sat below the fold."""
    for band in ("benign", "pre_cancerous", "malignant"):
        offered = [entry_id for entry_id, _q in suggested_questions(kb, band)]
        assert any(is_lesion_specific(i) for i in offered), f"{band}: {offered}"


def test_a_question_is_never_offered_twice(kb) -> None:
    offered = [entry_id for entry_id, _q in suggested_questions(kb, "general", n=12)]
    assert len(offered) == len(set(offered))


def test_excluded_questions_are_not_re_offered(kb) -> None:
    first = [entry_id for entry_id, _q in suggested_questions(kb, "malignant")]
    again = [
        entry_id
        for entry_id, _q in suggested_questions(kb, "malignant", exclude_ids=set(first))
    ]
    assert not set(first) & set(again)


def test_the_measured_sentence_quotes_the_real_numbers() -> None:
    sentence = signs_sentence(ABCDE)
    assert "17 mm" in sentence
    assert "two different colours" in sentence.lower()
    assert "uneven and blurred" in sentence.lower()


def test_no_measurements_means_no_invented_sentence() -> None:
    """A scan whose spot could not be outlined has nothing to report, and must
    not fall back to describing a typical lesion."""
    assert signs_sentence(None) == ""
    assert signs_sentence({}) == ""


def test_saved_scans_carry_their_own_measurements() -> None:
    """The case screen pins a stored row, not the live result. Its ABCDE lives
    in abcd_json / e_json, so the same sentence has to be reachable from there
    or a saved scan silently answers with no numbers."""
    stored = json.dumps({k: v for k, v in ABCDE.items() if k != "E"})
    revived = json.loads(stored)
    revived["E"] = json.loads(json.dumps(ABCDE["E"]))
    assert signs_sentence(revived) == signs_sentence(ABCDE)


def test_lesion_answers_never_reach_the_rephraser(monkeypatch) -> None:
    """reword()'s only content guard is a timeframe check, so nothing would
    catch "17 mm" coming back as "about two centimetres"."""
    import views.assistant_view as av

    called = {"n": 0}

    class _Boom:
        def reword(self, text):  # pragma: no cover - must never run
            called["n"] += 1
            return text, True

    monkeypatch.setattr(av, "OllamaRephraser", _Boom)
    monkeypatch.setattr(av.st, "session_state", {"assistant_gemma_enabled": True})

    class _Entry:
        id = "why_flagged_general"
        answer = "Because of the five checks."
        always_append_disclaimer = True

    class _Matcher:
        def match(self, _q, _band):
            return _Entry(), 1.0

    ctx = av.ScanContext(band="malignant", label="malignant", abcde=ABCDE)
    text, _disclaim, reworded = av._answer(object(), _Matcher(), "why?", ctx)

    assert called["n"] == 0, "a lesion-specific answer was sent to the rephraser"
    assert reworded is False
    assert "17 mm" in text
