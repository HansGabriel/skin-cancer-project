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
    assert v.tone == "success"
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
        assert "checked" in v.advice
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
