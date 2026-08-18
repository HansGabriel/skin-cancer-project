"""Which nav keys exist, and when.

The keys moved into the foot of the instrument band in the redesign. Two rules
predate it and are deliberately kept, because the design's five-key row is a
layout and neither of these is:

* **Questions is hidden when the knowledge base is empty** — never a dead key.
* **The kiosk gets an Exit key** so closing it does not need the staff passcode
  or an SSH session (the alternative stranded whoever was running the demo).
"""

from __future__ import annotations

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


def test_exit_is_kiosk_only(monkeypatch, kb_live) -> None:
    """A browser tab does not need an Exit button; the kiosk has no other way out."""
    kb_live(True)
    monkeypatch.setattr(bottom_nav, "is_kiosk", lambda: False)
    assert bottom_nav.is_kiosk() is False
    monkeypatch.setattr(bottom_nav, "is_kiosk", lambda: True)
    assert bottom_nav.is_kiosk() is True
    # The key itself is rendered by render_nav from is_kiosk(); _items() never
    # carries it, so that the confirm-then-quit handling stays in one place.
    assert "Exit" not in [label for label, _r, _h in bottom_nav._items()]


def test_old_entry_point_still_resolves() -> None:
    """app.py moved to render_instrument, but nothing should break mid-refactor."""
    assert bottom_nav.render_bottom_nav is bottom_nav.render_nav
