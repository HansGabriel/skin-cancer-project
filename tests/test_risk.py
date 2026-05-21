from __future__ import annotations

from services.abcde import LetterResult
from services.risk import composite_risk_score, evolution_weight, risk_band


def _abcde(e_verdict: str = "needs history") -> dict[str, LetterResult]:
    base = {"value": 0.1, "tier": 0, "verdict": "normal"}
    return {
        "A": base,
        "B": base,
        "C": base,
        "D": {**base, "value": 5.0},
        "E": {"value": None, "tier": 0, "verdict": e_verdict},
    }


def test_composite_low() -> None:
    score = composite_risk_score(10.0, _abcde())
    assert score < 34.0
    assert risk_band(score) == "low"


def test_evolution_changing_adds_weight() -> None:
    abcde = _abcde("changing")
    assert evolution_weight(abcde) == 1.0
    s = composite_risk_score(50.0, abcde)
    assert s > composite_risk_score(50.0, _abcde("stable"))
