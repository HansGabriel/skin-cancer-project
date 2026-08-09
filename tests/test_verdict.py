"""Single-verdict engine: exactly one coherent message per scan, never two.

The invariant that motivated the module: the old screen could render
"malignant / URGENT" and "Inconclusive" together. Here every reachable verdict
is checked to carry either urgency language or uncertainty language — never both.
"""

from __future__ import annotations

import itertools

from backend.contracts import ScanResult
from backend.recommendations import RECOMMENDATIONS
from services.verdict import (
    LOW_RISK_SAFETY_LINE,
    resolve_verdict,
    retake_verdict,
)

QUALITY_OK = {"ok": True, "reasons": [], "reason_details": []}
QUALITY_BAD = {
    "ok": False,
    "reasons": ["Image too blurry — please refocus and try again."],
    "reason_details": [("blur", "🔍 Out of focus", "warning")],
}

URGENT_WORDS = ("urgent", "1-2 weeks", "high concern")
UNSURE_WORDS = ("uncertain", "inconclusive", "not confident", "could not assess")


def _scan(label: str, confidence: float) -> ScanResult:
    return ScanResult(
        label=label,
        confidence=confidence,
        probs={"benign": 33.0, "pre_cancerous": 33.0, "malignant": 34.0},
        image_jpg_bytes=b"",
        timestamp="2026-08-08T00:00:00",
        inference_ms=100,
        urgency="",
        icon="",
        action="",
        backend_id="mock",
    )


def test_quality_fail_yields_retake_without_risk_language() -> None:
    v = resolve_verdict(None, QUALITY_BAD)
    assert v.state == "retake"
    assert QUALITY_BAD["reasons"][0] in v.reasons
    text = v.text().lower()
    assert not any(w in text for w in URGENT_WORDS)
    assert "concern" not in text and "risk" not in text


def test_retake_handles_missing_quality_dict() -> None:
    assert retake_verdict(None).state == "retake"
    assert resolve_verdict(None, None).state == "retake"


def test_benign_confident_is_low_concern_with_safety_line() -> None:
    v = resolve_verdict(_scan("benign", 80.0), QUALITY_OK)
    assert v.state == "low_concern"
    # Deliberately colourless: a clean screen must not be styled as an all-clear.
    assert v.tone == "neutral"
    assert LOW_RISK_SAFETY_LINE in v.advice


def test_benign_unconfident_is_uncertain() -> None:
    v = resolve_verdict(_scan("benign", 30.0), QUALITY_OK)
    assert v.state == "uncertain"
    assert v.tone == "info"
    assert LOW_RISK_SAFETY_LINE not in v.text()


def test_pre_cancerous_confident_needs_attention() -> None:
    v = resolve_verdict(_scan("pre_cancerous", 60.0), QUALITY_OK)
    assert v.state == "needs_attention"
    assert v.tone == "warning"
    assert RECOMMENDATIONS["pre_cancerous"]["action"] in v.advice


def test_malignant_confident_is_urgent() -> None:
    v = resolve_verdict(_scan("malignant", 70.0), QUALITY_OK)
    assert v.state == "urgent"
    assert v.tone == "urgent"
    assert RECOMMENDATIONS["malignant"]["action"] in v.advice


def test_flagged_but_unconfident_keeps_referral_without_urgent_styling() -> None:
    for label in ("malignant", "pre_cancerous"):
        v = resolve_verdict(_scan(label, 20.0), QUALITY_OK)
        assert v.state == "uncertain_caution"
        assert v.tone != "urgent"
        # Low confidence must not drop the referral — it only drops the urgency.
        assert "health worker" in v.advice
        assert "urgent" not in v.text().lower()


def test_confidence_floor_is_inclusive() -> None:
    # Exactly at the floor counts as confident (>=), matching the old banner rule.
    assert resolve_verdict(_scan("benign", 45.0), QUALITY_OK).state == "low_concern"
    assert resolve_verdict(_scan("benign", 44.9), QUALITY_OK).state == "uncertain"


def test_env_floor_override(monkeypatch) -> None:
    monkeypatch.setenv("SKIN_INCONCLUSIVE_BELOW", "60")
    assert resolve_verdict(_scan("benign", 50.0), QUALITY_OK).state == "uncertain"
    monkeypatch.setenv("SKIN_INCONCLUSIVE_BELOW", "not-a-number")
    assert resolve_verdict(_scan("benign", 50.0), QUALITY_OK).state == "low_concern"


def test_explicit_floor_argument_wins() -> None:
    v = resolve_verdict(_scan("malignant", 50.0), QUALITY_OK, confidence_floor=80.0)
    assert v.state == "uncertain_caution"


def test_cross_product_always_one_coherent_verdict() -> None:
    """Every reachable input produces exactly one verdict whose copy never mixes
    urgency language with uncertainty language — the old two-banner bug."""
    labels = ("benign", "pre_cancerous", "malignant")
    confidences = (5.0, 30.0, 44.9, 45.0, 60.0, 99.0)
    qualities = (QUALITY_OK, QUALITY_BAD, None)
    verdicts = [retake_verdict(QUALITY_BAD), resolve_verdict(None, QUALITY_BAD)]
    for label, conf, q in itertools.product(labels, confidences, qualities):
        verdicts.append(resolve_verdict(_scan(label, conf), q))
    for v in verdicts:
        assert v.state in {
            "retake",
            "low_concern",
            "uncertain",
            "needs_attention",
            "urgent",
            "uncertain_caution",
        }
        assert v.headline and v.body and v.advice
        text = v.text().lower()
        has_urgent = any(w in text for w in URGENT_WORDS)
        has_unsure = any(w in text for w in UNSURE_WORDS)
        assert not (has_urgent and has_unsure), f"{v.state} mixes urgency and uncertainty: {text}"


# --- One voice across every screen -------------------------------------------


class _SavedScan:
    """Minimal stand-in for services.storage.Scan (only what the verdict needs)."""

    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence


def test_saved_scan_says_the_same_thing_as_the_live_result() -> None:
    """History, case and assistant screens used to re-derive wording from the
    composite risk band, so one scan could read "URGENT" in a list and
    "NOT SURE" on its own screen. Both paths must resolve identically."""
    from services.verdict import verdict_for_saved_scan

    for label in ("benign", "pre_cancerous", "malignant"):
        for conf in (5.0, 30.0, 44.9, 45.0, 60.0, 99.0):
            live = resolve_verdict(_scan(label, conf), QUALITY_OK)
            saved = verdict_for_saved_scan(_SavedScan(label, conf))
            assert live.state == saved.state
            assert live.headline == saved.headline
            assert live.tone == saved.tone


def test_no_lesion_is_distinct_from_retake() -> None:
    """"I cannot read this photo" and "this is not a skin spot" are different
    problems; telling someone to fix the lighting when they photographed a wall
    is useless advice."""
    from services.verdict import no_lesion_verdict

    v = no_lesion_verdict(("No skin was found in this photo.",))
    assert v.state == "no_lesion"
    assert v.state != retake_verdict(None).state
    assert v.headline != retake_verdict(None).headline
    text = v.text().lower()
    assert not any(w in text for w in URGENT_WORDS)
    assert "concern" not in text and "risk" not in text


def test_no_lesion_keeps_the_reason_it_was_given() -> None:
    from services.verdict import no_lesion_verdict

    assert "No skin was found in this photo." in no_lesion_verdict(
        ["No skin was found in this photo."]
    ).reasons


def test_verdict_copy_stays_readable() -> None:
    """Plain words only: nothing a visitor reads may contain model jargon."""
    from services.verdict import no_lesion_verdict, verdict_for_label

    jargon = (
        "softmax", "logit", "inference", "classifier", "cnn", "model",
        "confidence floor", "threshold", "calibrat", "probability", "tensor",
        "malignant", "pre_cancerous", "benign",
    )
    verdicts = [retake_verdict(None), no_lesion_verdict()] + [
        verdict_for_label(label, conf)
        for label in ("benign", "pre_cancerous", "malignant")
        for conf in (10.0, 90.0)
    ]
    for v in verdicts:
        text = v.text().lower()
        for word in jargon:
            assert word not in text, f"{v.state} exposes '{word}': {v.text()}"
