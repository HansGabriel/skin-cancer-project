"""Composite risk score (Tier 1: ÷8 over A–D tiers). TODO: switch to ÷10 when E has real tiers (Notion Part 2.2)."""

from __future__ import annotations

from .abcde import LetterResult, abcde_tier_sum_ad


def composite_risk_score(
    p_malignant_percent: float,
    abcde: dict[str, LetterResult] | None,
) -> float:
    """Return 0–100 composite: 0.6 * P(malignant in 0–1) + 0.4 * (sum tiers A–D)/8.

    If ``abcde`` is ``None`` (no lesion mask), A–D tier sum is treated as 0.
    """
    p = max(0.0, min(100.0, p_malignant_percent)) / 100.0
    s = abcde_tier_sum_ad(abcde) if abcde is not None else 0
    return float(100.0 * (0.6 * p + 0.4 * (s / 8.0)))


def risk_band(score: float) -> str:
    if score < 34.0:
        return "low"
    if score <= 66.0:
        return "moderate"
    return "high"
