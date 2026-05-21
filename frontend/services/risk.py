"""Composite risk score with A–E tiers and evolution weight."""

from __future__ import annotations

from .abcde import LetterResult


def abcde_tier_sum_all(abcde: dict[str, LetterResult] | None) -> int:
    if abcde is None:
        return 0
    return int(sum(abcde[k]["tier"] for k in ("A", "B", "C", "D", "E") if k in abcde))


def evolution_weight(abcde: dict[str, LetterResult] | None) -> float:
    if abcde is None:
        return 0.0
    e = abcde.get("E", {})
    verdict = str(e.get("verdict", ""))
    if verdict in ("stable", "needs history"):
        return 0.0
    if verdict == "watch":
        return 0.5
    if verdict == "changing":
        return 1.0
    return 0.0


def composite_risk_score(
    p_malignant_percent: float,
    abcde: dict[str, LetterResult] | None,
) -> float:
    """0.55·p + 0.35·(sum tiers A–E)/10 + 0.10·evolution_weight."""
    p = max(0.0, min(100.0, p_malignant_percent)) / 100.0
    s = abcde_tier_sum_all(abcde)
    ev = evolution_weight(abcde)
    return float(100.0 * (0.55 * p + 0.35 * (s / 10.0) + 0.10 * ev))


def risk_band(score: float) -> str:
    if score < 34.0:
        return "low"
    if score <= 66.0:
        return "moderate"
    return "high"
