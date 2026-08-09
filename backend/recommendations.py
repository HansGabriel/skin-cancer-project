"""Display strings keyed by collapsed diagnosis label (Streamlit / PC side).

Keys must match exactly one line per row in ``models/labels.txt`` (3-class deployment).
"""

from __future__ import annotations

from typing import TypedDict


class Recommendation(TypedDict):
    icon: str
    urgency: str
    action: str


RECOMMENDATIONS: dict[str, Recommendation] = {
    "benign": {
        "icon": "🟢",
        "urgency": "LOW CONCERN",
        "action": "No action needed now. Keep checking your skin from time to time.",
    },
    "pre_cancerous": {
        "icon": "🟠",
        "urgency": "IMPORTANT",
        "action": "See a skin doctor within a month.",
    },
    "malignant": {
        "icon": "🔴",
        "urgency": "URGENT",
        "action": "See a skin doctor within one to two weeks.",
    },
}

# The labels that mean "the screen flagged this". One definition, imported by
# services.verdict and scripts/eval_threshold.py — keeping labels, metrics and
# UI in sync is an AGENTS.md rule, and three private copies is how they drift.
CANCER_LABELS: tuple[str, ...] = ("pre_cancerous", "malignant")
