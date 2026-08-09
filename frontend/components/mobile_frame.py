"""The page-width seam.

This used to emit ``<div class="ds-mobile-frame">`` around a screen's contents.
It never worked: Streamlit closes the HTML in every ``st.markdown`` call, so the
opening and closing tags became two separate empty divs and nothing was ever
wrapped — the frame's ``max-width`` had no effect, and each screen carried two
phantom 16px elements.

Page width now lives on ``.block-container`` in ``theme/css.py``, which is the
only element Streamlit lets us size. The context manager is kept because every
screen marks its body with it, and that is still the right seam to hold any
future per-screen layout — but it deliberately emits nothing.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def mobile_frame() -> Iterator[None]:
    yield
