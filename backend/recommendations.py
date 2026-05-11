"""Display strings keyed by collapsed diagnosis label (Streamlit / PC side)."""

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
        "action": "No immediate action needed. Continue regular self-checks.",
    },
    "pre_cancerous": {
        "icon": "🟠",
        "urgency": "IMPORTANT",
        "action": "Schedule a dermatology consultation within 1 month.",
    },
    "malignant": {
        "icon": "🔴",
        "urgency": "URGENT",
        "action": "See a dermatologist within 1-2 weeks.",
    },
}
