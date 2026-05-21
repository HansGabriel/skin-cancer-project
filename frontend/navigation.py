"""Session-state screen router."""

from __future__ import annotations

from typing import Literal

import streamlit as st

Route = Literal["home", "camera", "results", "history", "folder", "case", "settings"]
ROUTES: tuple[Route, ...] = ("home", "camera", "results", "history", "folder", "case", "settings")


def init_navigation() -> None:
    if "route" not in st.session_state:
        st.session_state["route"] = "home"


def navigate(route: Route, *, rerun: bool = True, **session_updates: object) -> None:
    st.session_state["route"] = route
    for key, value in session_updates.items():
        st.session_state[key] = value
    if rerun:
        st.rerun()


def current_route() -> Route:
    init_navigation()
    route = st.session_state.get("route", "home")
    return route if route in ROUTES else "home"  # type: ignore[return-value]
