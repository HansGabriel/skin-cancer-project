"""The Questions tab: written answers only, plain words, matched to the visitor's result."""

from __future__ import annotations

import json
import re

import pytest

from dermascan import answers, verdict
from test_verdict import JARGON


@pytest.fixture(scope="module")
def bank():
    return answers.AnswerBank()


def test_every_answer_is_plain_language(bank) -> None:
    raw = json.loads(answers.ANSWERS_PATH.read_text())
    texts = [raw["fallback_answer"]] + [e.answer for e in bank.entries] + [e.web_answer for e in bank.entries if e.web_answer]
    texts += [q for e in bank.entries for q in e.questions]
    for text in texts:
        for word in JARGON:
            assert not re.search(rf"\b{re.escape(word)}\b", text.lower()), f"'{word}' in {text!r}"
        assert "%" not in text and not re.search(r"\d", text), f"a number in {text!r}"


def test_answers_keep_the_verdict_timeframes(bank) -> None:
    by_id = {e.id: e.answer for e in bank.entries}
    assert "within a month" in by_id["checked_what_now"]
    assert "within one to two weeks" in by_id["urgent_what_now"]
    assert verdict.ACTION["malignant"].rstrip(".").lower().endswith("within one to two weeks")


@pytest.mark.parametrize(
    ("question", "state", "expected"),
    [
        ("what does my result mean", "low_concern", "clean_meaning"),
        ("what does my result mean", "urgent", "urgent_meaning"),
        ("who can see my photo?", None, "general_privacy"),
        ("how big is it", "needs_attention", "about_size"),
        ("why is it not sure", "uncertain", "unsure_meaning"),
    ],
)
def test_questions_find_their_answer(bank, question, state, expected) -> None:
    assert bank.ask(question, state).entry_id == expected


def test_a_result_never_gets_another_results_answer(bank) -> None:
    a = bank.ask("what does my result mean", "low_concern")
    assert not a.entry_id.startswith(("urgent", "checked"))


def test_off_topic_gets_the_fallback(bank) -> None:
    a = bank.ask("best pizza in town", "urgent")
    assert a.entry_id is None and a.text == bank.fallback and a.suggestions


def test_about_spot_answers_carry_the_visitors_own_signs(bank) -> None:
    a = bank.ask("which warning signs did it find", "needs_attention", ["Edges are slightly uneven"])
    assert a.text.endswith("On your photo: edges are slightly uneven.")


def test_unreviewed_answers_say_so(bank) -> None:
    a = bank.ask("is this a diagnosis")
    assert a.reviewed_by is None and a.to_dict()["reviewed"] is False


def test_a_drafted_answer_names_its_drafter_until_a_doctor_signs_it(tmp_path) -> None:
    """scripts/draft_answers_gemma.py sets drafted_by; the badge says so, and the drafter shows in the intro."""
    raw = json.loads(answers.ANSWERS_PATH.read_text())
    raw["entries"][0]["drafted_by"] = "Gemma 4 e4b, 2026-11-02"
    raw["entries"][1].update(drafted_by="Gemma 4 e4b, 2026-11-02", reviewed_by="Dr Example", reviewed_date="2026-11-03")
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(raw))
    bank = answers.AnswerBank(path)
    drafted = bank.ask(raw["entries"][0]["questions"][0], "low_concern").to_dict()
    assert drafted["badge"] == "DRAFTED BY GEMMA 4 E4B · NOT YET REVIEWED BY A DOCTOR" and drafted["reviewed"] is False
    signed = bank.ask(raw["entries"][1]["questions"][0], "low_concern").to_dict()
    assert signed["badge"] == "REVIEWED BY DR EXAMPLE" and signed["reviewed"] is True
    assert bank.drafters() == ["Gemma 4 e4b"]
    assert answers.AnswerBank().drafters() == []  # the shipped file: team-written until the script runs


def test_suggestions_put_the_results_own_questions_first(bank) -> None:
    s = bank.suggestions("urgent")
    assert s[0] == "What does see a doctor soon mean?" and len(s) == answers.config.SUGGESTED_QUESTIONS
    assert all("my" not in q.lower().split() for q in bank.suggestions("general"))


def test_every_verdict_state_has_a_band() -> None:
    assert set(verdict.CHIP) == set(answers.BAND_FOR_STATE)


def test_blank_sign_lines_do_not_break_an_answer(bank) -> None:
    a = bank.ask("which warning signs did it find", "urgent", ["", "  "])
    assert "On your photo" not in a.text and a.to_dict()["badge"].startswith("WRITTEN BY THE PROJECT TEAM")
