"""The pinned actions row — created beside the scroll box, filled from inside it.

The two or three buttons at the foot of a screen used to be the last child of
``.st-key-epv-body``, held down by ``margin-top:auto``. That put them *inside*
the scrolling region, which caused the bug where the results screen opened
already scrolled past its own headline: after a rerun the browser scrolls the
element that has focus back into view, and the element with focus is whichever
button was just tapped — at the very bottom of a 700px-tall screen inside a
521px box. The panel obediently scrolled to the bottom.

Streamlit will not let a view reach out of the container it is running in, and
``:has()`` is banned here (WebKitGTK), so the row cannot be lifted out with CSS
either. What Streamlit *does* allow is filling a container that was created
earlier. So ``app.py`` creates the row as a sibling of the scroll box before it
dispatches the screen, and views write into it through ``actions_slot()``.

The slot is consumed once per run. A view called outside the shell — in a test,
say — gets its own container and behaves exactly as before.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st

_SLOT_KEY = "_epv_actions_slot"


def open_actions_slot():
    """Create the row in DOM order and register it for this run.

    Called once per run by ``app.py``, between the scroll box and the safety
    strip. Registering overwrites any slot left behind by a run that stopped
    early, so a stale container can never be written into.
    """
    slot = st.container(key="epv-actions")
    st.session_state[_SLOT_KEY] = slot
    return slot


@contextmanager
def actions_slot() -> Iterator[None]:
    """Write the screen's pinned actions into the row app.py laid out."""
    slot = st.session_state.pop(_SLOT_KEY, None)
    if slot is None:
        slot = st.container(key="epv-actions")
    with slot:
        yield
