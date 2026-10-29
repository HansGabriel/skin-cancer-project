"""One coherent message per scan, in words a visitor can act on."""

from __future__ import annotations

import re

from dermascan import verdict
from dermascan.classifier import Prediction
from dermascan.gate import Refusal

JARGON = ("malignant", "benign", "softmax", "inconclusive", "pre_cancerous", "classifier", "model", "tflite")
URGENT = ("urgent", "doctor soon")


def _pred(label: str, conf: float) -> Prediction:
    return Prediction(label, conf, {"benign": 1, "pre_cancerous": 1, "malignant": 1}, (0.3, 0.3, 0.4), 10, label != "benign")


def test_every_verdict_is_plain_language() -> None:
    cases = [verdict.for_prediction(_pred(l, c)) for l in ("benign", "pre_cancerous", "malignant") for c in (20.0, 80.0)]
    cases += [verdict.for_refusal(Refusal(code, True)) for code in list(verdict._REFUSAL_COPY) + ["unknown"]]
    cases += [verdict.error_verdict(k) for k in list(verdict._ERROR_BODY) + ["unknown"]]
    cases.append(verdict.cannot_judge())
    for v in cases:
        text = v.text().lower()
        for word in JARGON:
            assert not re.search(rf"\b{re.escape(word)}\b", text), f"{v.state}: '{word}' in {text!r}"
        assert "%" not in text and not re.search(r"\d", text), f"{v.state}: a number or percentage in {text!r}"
        assert v.headline.isupper()
    for word in JARGON:
        assert word not in verdict.FORCED_CAVEAT.lower()


def test_clean_result_has_no_colour_and_keeps_the_safety_line() -> None:
    v = verdict.for_prediction(_pred("benign", 80.0))
    assert v.state == "low_concern" and v.tone == "neutral"
    assert verdict.LOW_RISK_SAFETY_LINE in v.text()
    for sign in ("changes", "grows", "bleeds"):
        assert sign in verdict.LOW_RISK_SAFETY_LINE


def test_unsure_benign_is_not_reassuring() -> None:
    v = verdict.for_prediction(_pred("benign", 30.0))
    assert v.state == "uncertain"
    assert verdict.LOW_RISK_SAFETY_LINE not in v.text()


def test_flagged_and_confident_sends_to_a_doctor() -> None:
    assert verdict.for_prediction(_pred("malignant", 70.0)).state == "urgent"
    assert verdict.for_prediction(_pred("pre_cancerous", 60.0)).state == "needs_attention"


def test_flagged_but_unsure_keeps_the_referral_without_urgency() -> None:
    for label in ("malignant", "pre_cancerous"):
        v = verdict.for_prediction(_pred(label, 20.0))
        assert v.state == "uncertain_caution" and v.tone != "urgent"
        assert "health worker" in v.advice
        assert not any(w in v.text().lower() for w in URGENT)


def test_cannot_judge_gives_no_verdict_and_no_reassurance() -> None:
    v = verdict.cannot_judge()
    assert v.state == "cannot_judge" and v.tone == "info" and verdict.CHIP[v.state] == "NOT READ"
    assert "health worker" in v.advice
    for w in ("safe", "clear", "nothing stood out"):
        assert w not in v.text().lower()


def test_confidence_floor_is_inclusive() -> None:
    assert verdict.for_prediction(_pred("benign", 45.0)).state == "low_concern"
    assert verdict.for_prediction(_pred("benign", 44.9)).state == "uncertain"


def test_each_refusal_code_has_its_own_words() -> None:
    seen = {verdict.for_refusal(Refusal(c, True)).headline for c in verdict._REFUSAL_COPY}
    assert len(seen) == len(verdict._REFUSAL_COPY)
    assert verdict.for_refusal(Refusal("something_new", True)).headline == "NOT A SPOT IT CAN READ"


def _plain(text: str) -> None:
    for word in JARGON:
        assert not re.search(rf"\b{re.escape(word)}\b", text.lower()), f"'{word}' in {text!r}"
    assert "%" not in text and not re.search(r"\d", text), f"a number in {text!r}"


def test_sign_lines_are_plain_and_skip_what_was_not_measured() -> None:
    from dermascan.signs import Sign

    for tier in (0, 1, 2):
        lines = verdict.sign_lines([Sign(l, 0.1, tier) for l in "ABC"] + [Sign("D", 0.2, None), Sign("E", None, None)])
        assert [x["letter"] for x in lines] == ["A", "B", "C"]
        for x in lines:
            _plain(x["text"])
    assert verdict.sign_lines(None) == []


def test_photo_readings_cover_every_refusal() -> None:
    full = {"brightness": 120.0, "focus": 80.0, "spot_fraction": 0.2}
    assert [r["level"] for r in verdict.photo_readings(full, None)] == [3, 3, 3]
    assert verdict.photo_readings({"brightness": 30.0, "focus": 30.0, "spot_fraction": 0.2}, None)[0]["word"] == "dim"
    for code in list(verdict._REFUSAL_COPY):
        rows = verdict.photo_readings({"skin_fraction": 0.5}, code)
        assert any(r["level"] is not None and r["level"] < 3 for r in rows), code
        for r in rows:
            _plain(r["word"])
    for text in (verdict.PHOTO_OK_HEADLINE, verdict.PHOTO_OK_LEDE, verdict.PHOTO_SOFT_LEDE, *verdict.CHIP.values()):
        _plain(text)
    assert "SAFE" not in " ".join(verdict.CHIP.values())


def test_heat_words_are_plain_and_never_a_verdict() -> None:
    """The lines beside the heatmap circle: plain, class-agnostic, the only nudge is to retake."""
    assert set(verdict.HEAT_WORDS) == {"loading", "on", "off", "unknown"}
    for text in verdict.HEAT_WORDS.values():
        _plain(text)
        assert len(text) <= 110, text  # two lines beside the result's small circle at 1024 x 600
        for word in ("cancer", "safe", "normal", "why"):
            assert word not in text.lower(), text
    assert "retake" in verdict.HEAT_WORDS["off"].lower()


def test_every_check_outcome_has_a_short_plain_label() -> None:
    """The chip on the check screen: NOT A SPOT · ..., SPOT · ..., CAN'T READ · ..., or SPOT FOUND."""
    from dermascan.gate import FRAMING_CODES, NOT_A_SPOT_CODES

    assert verdict.check_label("pass", None) == "SPOT FOUND"
    assert verdict.check_label("error", None) == verdict.check_label("refused", None) == "NOT READ"
    assert verdict.check_label("refused", "something_new") == "NOT A SPOT"
    for code in verdict._REFUSAL_COPY:
        label = verdict.check_label("refused", code)
        _plain(label)
        assert label.isupper() and len(label) <= 28, label
        if code in NOT_A_SPOT_CODES:
            assert label.startswith("NOT A SPOT · "), code
        elif code in FRAMING_CODES:
            assert label.startswith("SPOT · "), code  # a real spot: "Check it anyway" still answers
        elif code == "no_skin":
            assert label == "SKIN NOT RECOGNISED"
        else:
            assert label.startswith("CAN'T READ · "), code
    assert set(verdict._REFUSAL_COPY) <= set(verdict.CHECK_LABEL)
    _plain(verdict.PHOTO_CHECK_EYEBROW)
