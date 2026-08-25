"""Session-state screen router."""

from __future__ import annotations

from typing import Literal

import streamlit as st

Route = Literal[
    "home",
    "camera",
    "reading",
    "results",
    "history",
    "folder",
    "case",
    "settings",
    "assistant",
    "staff",
]
ROUTES: tuple[Route, ...] = (
    "home",
    "camera",
    "reading",
    "results",
    "history",
    "folder",
    "case",
    "settings",
    "assistant",
    "staff",
)


def init_navigation() -> None:
    if "route" not in st.session_state:
        st.session_state["route"] = "home"


# Routes that legitimately keep working with the scan that is on screen: the
# assistant answers questions *about* it, the staff readout shows its numbers,
# the reading screen is where it is produced, and the case screen is where the
# save flow lands. Every other destination is someone walking away from their
# result.
_KEEPS_RESULT: tuple[Route, ...] = ("results", "assistant", "case", "staff", "reading")


def _discard_result_on_leave(target: Route) -> None:
    """Drop the participant's photo when leaving the results screen.

    Enforced here rather than in each button because the leak came back exactly
    that way: "Done" cleared it, but the bottom navigation called navigate()
    straight, so tapping Home or Saved from a result left the previous
    participant's skin in session state, one Back tap from the next visitor.
    docs/PRIVACY.md promises this, so it has to hold for every route change.
    """
    if st.session_state.get("route") in _KEEPS_RESULT and target not in _KEEPS_RESULT:
        st.session_state.pop("last_result", None)
        # The capture bytes outlive a refused scan on purpose (the result
        # screen offers "Check it anyway"), so they have to be dropped on the
        # same boundary or the next visitor lands in step 2 on the previous
        # visitor's photo.
        st.session_state.pop("capture_image_bytes", None)
        _drop_photo_caches()


# The live preview is the most expensive thing the device does, and only one
# screen shows it.
_NEEDS_CAMERA: tuple[Route, ...] = ("camera",)


def _release_camera_on_leave(target: Route) -> None:
    """Hand the camera back when the screen that needs it is left behind."""
    if st.session_state.get("route") in _NEEDS_CAMERA and target not in _NEEDS_CAMERA:
        from services.pi_camera import release_camera

        release_camera()


def _drop_photo_caches() -> None:
    """Forget every memoised copy of the photo.

    The capture readings and the aperture thumbnail are both memoised on the
    raw JPEG bytes, which means an lru_cache holds a participant's skin after
    session state has dropped it. docs/PRIVACY.md promises otherwise, so the
    caches clear on the same boundary.

    Delegated to ``services.photo_cache`` rather than importing the two cache
    owners by name. The router has no business knowing which components memoise
    a capture, and the "Take another" buttons inside the camera view need the
    same call without going near this module.
    """
    from services.photo_cache import forget_photos

    forget_photos()


def navigate(route: Route, *, rerun: bool = True, **session_updates: object) -> None:
    _release_camera_on_leave(route)
    _discard_result_on_leave(route)
    st.session_state["route"] = route
    for key, value in session_updates.items():
        st.session_state[key] = value
    if rerun:
        st.rerun()


def current_route() -> Route:
    init_navigation()
    route = st.session_state.get("route", "home")
    return route if route in ROUTES else "home"  # type: ignore[return-value]
