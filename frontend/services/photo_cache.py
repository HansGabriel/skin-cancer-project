"""One place to forget every memoised copy of a participant's photo.

Two caches hold decoded or re-encoded capture bytes so that a rerun does not
redo the work: the aperture thumbnail (``components.aperture``) and the step-2
capture readings (``views.camera_view``). Both are keyed on the raw JPEG, which
means both hold a participant's skin for as long as they hold an entry.

``docs/PRIVACY.md`` promises that *"Every exit from the Results screen — 'Done',
'Clear', 'Take another photo', or a reboot — drops the photo from session
state"*. Clearing was originally wired into ``navigation`` alone, which covers
route changes but **not** the two "Take another" buttons inside the camera view:
those drop the bytes and rerun without navigating anywhere, so the caches kept
the discarded photo. This module exists so every one of those boundaries can
call one function, and so a new cache is registered once rather than remembered
in the router.

Note these are ``functools.lru_cache`` objects on module-level functions, so
they are per *process*, not per session. Nothing can be read out of them without
the exact bytes that produced the entry, so this is not a disclosure path
between concurrent web-app sessions — but it is still a copy of a photo
outliving the session that made it, which is what the promise is about.
"""

from __future__ import annotations

from typing import Callable

_CLEARERS: list[Callable[[], None]] = []


def register(clear: Callable[[], None]) -> None:
    """Add a cache's clear function. Called at import by each cache's owner."""
    if clear not in _CLEARERS:
        _CLEARERS.append(clear)


def forget_photos() -> None:
    """Drop every cached copy of a capture. Safe to call at any boundary."""
    for clear in _CLEARERS:
        clear()
