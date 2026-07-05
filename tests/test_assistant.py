"""Assistant KB loading, review gating, TF-IDF matching, and Ollama fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from backend.assistant import (
    Matcher,
    OllamaRephraser,
    SIM_THRESHOLD,
    load_kb,
    suggested_questions,
)

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "assistant_kb.json"


def _write_kb(tmp_path: Path, entries: list[dict], fallback: str = "fallback") -> Path:
    p = tmp_path / "kb.json"
    p.write_text(json.dumps({"version": 1, "fallback_answer": fallback, "entries": entries}))
    return p


def _entry(**over) -> dict:
    base = {
        "id": "e1",
        "result_band": "general",
        "questions": ["What is melanoma?"],
        "answer": "An answer.",
        "always_append_disclaimer": True,
        "reviewed_by": "Dr. X",
        "reviewed_date": "2026-07-01",
    }
    base.update(over)
    return base


# ── KB loading / review gating ───────────────────────────────────────────────


def test_shipped_kb_is_valid_schema(monkeypatch):
    monkeypatch.setenv("SKIN_KB_DEV", "1")
    kb = load_kb(KB_PATH)
    assert kb.entries, "shipped KB should have entries in dev mode"
    assert kb.fallback_answer


def test_unreviewed_entries_hidden_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SKIN_KB_DEV", raising=False)
    p = _write_kb(tmp_path, [_entry(reviewed_by=None, reviewed_date=None)])
    assert load_kb(p).entries == ()


def test_unreviewed_entries_visible_in_dev(tmp_path, monkeypatch):
    monkeypatch.setenv("SKIN_KB_DEV", "1")
    p = _write_kb(tmp_path, [_entry(reviewed_by=None, reviewed_date=None)])
    assert len(load_kb(p).entries) == 1


def test_reviewed_entries_always_visible(tmp_path, monkeypatch):
    monkeypatch.delenv("SKIN_KB_DEV", raising=False)
    p = _write_kb(tmp_path, [_entry()])
    assert len(load_kb(p).entries) == 1


def test_bad_band_raises(tmp_path):
    p = _write_kb(tmp_path, [_entry(result_band="bogus")])
    with pytest.raises(ValueError, match="unknown result_band"):
        load_kb(p)


def test_duplicate_id_raises(tmp_path):
    p = _write_kb(tmp_path, [_entry(), _entry()])
    with pytest.raises(ValueError, match="duplicate"):
        load_kb(p)


def test_missing_field_raises(tmp_path):
    e = _entry()
    del e["answer"]
    p = _write_kb(tmp_path, [e])
    with pytest.raises(ValueError, match="missing required field"):
        load_kb(p)


# ── Matching ─────────────────────────────────────────────────────────────────


@pytest.fixture
def kb(monkeypatch):
    monkeypatch.setenv("SKIN_KB_DEV", "1")
    return load_kb(KB_PATH)


def test_paraphrase_matches_right_entry(kb):
    m = Matcher(kb)
    entry, score = m.match("what exactly does a malignant result mean", "malignant")
    assert entry is not None and entry.id == "malignant_meaning"
    assert score >= SIM_THRESHOLD


def test_gibberish_returns_none(kb):
    entry, _ = Matcher(kb).match("zxqv flurble wombat", "benign")
    assert entry is None


def test_band_filtering(kb):
    # A malignant-band query must never surface a benign-only entry.
    entry, _ = Matcher(kb).match("do I still need to see a doctor", "malignant")
    assert entry is None or entry.result_band in ("malignant", "general")


def test_general_entries_reachable_from_any_band(kb):
    entry, _ = Matcher(kb).match("what is the ABCDE rule", "benign")
    assert entry is not None and entry.id == "general_abcde"


def test_suggested_questions_band_first(kb):
    sugg = suggested_questions(kb, "malignant")
    assert sugg, "suggestions should exist"
    first_entry = next(e for e in kb.entries if e.id == sugg[0][0])
    assert first_entry.result_band == "malignant"


# ── Ollama layer: every failure returns verbatim text ────────────────────────


APPROVED = "See a dermatologist within 1-2 weeks. This is a screening aid."


def test_reword_timeout_falls_back():
    r = OllamaRephraser()
    with mock.patch.object(r, "_generate", return_value=None):
        text, reworded = r.reword(APPROVED)
    assert text == APPROVED and reworded is False


def test_reword_guard_rejects_dropped_timeframe():
    r = OllamaRephraser()
    with mock.patch.object(r, "_generate", return_value="Please go see a skin doctor soon."):
        text, reworded = r.reword(APPROVED)
    assert text == APPROVED and reworded is False  # "within 1-2 weeks" was dropped


def test_reword_guard_rejects_bloat():
    r = OllamaRephraser()
    with mock.patch.object(r, "_generate", return_value="waffle " * 200):
        text, reworded = r.reword(APPROVED)
    assert text == APPROVED and reworded is False


def test_reword_accepts_fact_preserving_output():
    good = "You should see a dermatologist within 1-2 weeks — remember this is only a screening aid."
    r = OllamaRephraser()
    with mock.patch.object(r, "_generate", return_value=good):
        text, reworded = r.reword(APPROVED)
    assert text == good and reworded is True


def test_pick_followups_rejects_out_of_range():
    r = OllamaRephraser()
    with mock.patch.object(r, "_generate", return_value="7, 9"):
        assert r.pick_followups(["a", "b", "c"]) is None


def test_pick_followups_parses_indices():
    r = OllamaRephraser()
    with mock.patch.object(r, "_generate", return_value="2, 0"):
        assert r.pick_followups(["a", "b", "c"]) == ["c", "a"]


def test_health_error_when_no_server(monkeypatch):
    r = OllamaRephraser(base_url="http://127.0.0.1:1")  # nothing listens here
    assert r.health()["status"] == "error"
