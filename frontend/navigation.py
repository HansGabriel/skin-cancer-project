"""Session-state screen router."""

from __future__ import annotations

from typing import Literal

import streamlit as st

Route = Literal["home", "camera", "results", "history", "folder", "case", "settings", "assistant"]
ROUTES: tuple[Route, ...] = ("home", "camera", "results", "history", "folder", "case", "settings", "assistant")


def init_navigation() -> None:
    if "route" not in st.session_state:
        st.session_state["route"] = "home"


# Routes that legitimately keep working with the scan that is on screen: the
# assistant answers questions *about* it, and the case screen is where the save
# flow lands. Every other destination is someone walking away from their result.
_KEEPS_RESULT: tuple[Route, ...] = ("results", "assistant", "case")


def _discard_result_on_leave(target: Route) -> None:
    """Drop the participant's photo when leaving the results screen.

    Enforced here rather than in each button because the leak came back exactly
    that way: "Done" cleared it, but the bottom navigation called navigate()
    straight, so tapping Home or Saved from a result left the previous
    participant's skin in session state, one Back tap from the next visitor.
    docs/PRIVACY.md promises this, so it has to hold for every route change.
    """
    if st.session_state.get("route") == "results" and target not in _KEEPS_RESULT:
        st.session_state.pop("last_result", None)


def navigate(route: Route, *, rerun: bool = True, **session_updates: object) -> None:
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
