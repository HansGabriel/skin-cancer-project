"""Guard: dev jargon and env-var names must not appear in user-facing strings.

The ONLY allowed home for env-var names / paths in the frontend is the
"Advanced (developer)" expander in settings_view.py.
"""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"

# Files whose user-visible strings must never mention env vars / dev concepts.
USER_FACING = [
    FRONTEND / "components" / "prob_bars.py",
    FRONTEND / "components" / "image_compare.py",
    FRONTEND / "components" / "abcde_row.py",
    FRONTEND / "views" / "assistant_view.py",
    FRONTEND / "views" / "camera_view.py",
]

# Regexes so internal identifiers (e.g. the SKIN_KERAS_PATH_UI session key) don't
# false-positive — we only care about the bare names as they'd appear in UI text.
FORBIDDEN = [
    r"SKIN_KERAS_PATH(?!_UI)",  # env var leaked in old differential expander
    r"SKIN_KB_DEV",  # env var leaked in old assistant empty-state
    r"reviewed_by",  # KB schema field leaked in old assistant empty-state
    r"assistant_kb\.json",  # internal filename
    r"7-class head",  # model-architecture jargon
    r"WSL",  # dev-environment jargon in camera caption
]


def test_no_dev_jargon_in_user_facing_files():
    offenders = []
    for f in USER_FACING:
        text = f.read_text(encoding="utf-8")
        for bad in FORBIDDEN:
            if re.search(bad, text):
                offenders.append(f"{f.name}: matches '{bad}'")
    assert not offenders, "\n".join(offenders)
