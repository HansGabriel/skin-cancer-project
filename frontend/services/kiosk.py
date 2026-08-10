"""Kiosk-mode detection — the one place the SKIN_KIOSK env contract lives."""

from __future__ import annotations

import os
from pathlib import Path

# launch_kiosk.sh polls for this file and tears the session down when it
# appears. Defined here so the app has one definition of the quit contract
# rather than the path being spelled out at each call site.
QUIT_FLAG = Path("/tmp/dermascan_quit")


def is_kiosk() -> bool:
    """True when running as the supervised Pi kiosk (set by launch_kiosk.sh)."""
    return os.environ.get("SKIN_KIOSK") == "1"


def request_quit() -> None:
    """Ask launch_kiosk.sh to shut the kiosk down.

    Best-effort: if /tmp is unwritable there is nothing useful to tell a
    visitor, and raising here would replace the app with a traceback on the
    one screen that has no keyboard to recover from it.
    """
    try:
        QUIT_FLAG.write_text("quit")
    except OSError:
        pass
