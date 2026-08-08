"""Kiosk-mode detection — the one place the SKIN_KIOSK env contract lives."""

from __future__ import annotations

import os


def is_kiosk() -> bool:
    """True when running as the supervised Pi kiosk (set by launch_kiosk.sh)."""
    return os.environ.get("SKIN_KIOSK") == "1"
