"""Which nav keys exist, and when.

The keys moved into the foot of the instrument band in the redesign. Two rules
predate it and are deliberately kept, because the design's five-key row is a
layout and neither of these is:

* **Questions is hidden when the knowledge base is empty** — never a dead key.
* **Exit is not a nav key at all.** It lives in Settings, behind the staff
  passcode. A sixth key left ~77px per key on the 1024px panel, which is
  narrower than "New check" renders, and it was the one button in the app
  carrying ``help=`` — which in Streamlit 1.57 wraps the ``<button>`` in two
  extra divs and unstyles it completely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from components import bottom_nav
from navigation import ROUTES


@pytest.fixture
def kb_live(monkeypatch):
    def _set(live: bool) -> None:
        monkeypatch.setattr(bottom_nav, "kb_is_live", lambda: live)

    return _set


def test_five_keys_when_the_knowledge_base_has_answers(kb_live) -> None:
    kb_live(True)
    labels = [label for label, _route, _here in bottom_nav._items()]
    assert labels == ["Home", "New check", "Saved", "Questions", "Settings"]


def test_questions_disappears_when_there_are_no_reviewed_answers(kb_live) -> None:
    kb_live(False)
    labels = [label for label, _route, _here in bottom_nav._items()]
    assert "Questions" not in labels
    assert labels == ["Home", "New check", "Saved", "Settings"]


def test_every_key_points_at_a_real_route(kb_live) -> None:
    kb_live(True)
    for _label, route, here in bottom_nav._items():
        assert route in ROUTES, route
        for r in here:
            assert r in ROUTES, r


def test_the_whole_capture_flow_lights_the_new_check_key(kb_live) -> None:
    """Frame, check, reading, result and the staff readout are one destination
    as far as a visitor is concerned, even though they are five screens."""
    kb_live(True)
    here = dict((route, here) for _l, route, here in bottom_nav._items())["camera"]
    for route in ("camera", "reading", "results", "staff"):
        assert route in here


def test_saved_key_covers_its_drill_down(kb_live) -> None:
    kb_live(True)
    here = dict((route, here) for _l, route, here in bottom_nav._items())["history"]
    for route in ("history", "folder", "case"):
        assert route in here


def test_exit_is_never_a_nav_key(kb_live) -> None:
    """The nav is at most five keys wide, in every configuration.

    At 1024px the instrument band is 461px. Five keys is 92px each; six was
    77px, and "New check" needs ~88px — which is what put "New checkSaved" on
    the panel. Nothing may add a sixth key.
    """
    for live in (True, False):
        kb_live(live)
        labels = [label for label, _r, _h in bottom_nav._items()]
        assert "Exit" not in labels
        assert len(labels) <= 5

    # And the module no longer reaches for kiosk state at all — the quit path
    # lives entirely in views/settings_view.py.
    assert not hasattr(bottom_nav, "is_kiosk")
    assert not hasattr(bottom_nav, "request_quit")


def test_the_kiosk_can_still_be_closed() -> None:
    """Removing the nav key must not strand whoever is running the demo."""
    source = (
        Path(bottom_nav.__file__).resolve().parents[1] / "views" / "settings_view.py"
    ).read_text()
    assert "request_quit()" in source


def test_old_entry_point_still_resolves() -> None:
    """app.py moved to render_instrument, but nothing should break mid-refactor."""
    assert bottom_nav.render_bottom_nav is bottom_nav.render_nav
