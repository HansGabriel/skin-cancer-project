"""One place that owns every Settings value: its default, its env fallback, and
whether it survives a restart.

This module exists because of a defect, and the defect is worth stating so it is
not reintroduced.

Every control on the Settings screen used to be a keyed Streamlit widget whose
``key`` *was* the canonical ``session_state`` key — ``strict_checking``,
``tta_toggle``, ``pixels_per_mm_ui`` and the rest. Streamlit garbage-collects a
keyed widget's ``session_state`` entry on any run where that widget is not
rendered, and ``navigation.navigate()`` makes every screen change a full
``st.rerun()`` on which the Settings screen is not drawn. So the value was
dropped, and ``app._init_session`` politely re-seeded the default.

The visible half of that was a toggle that would not stay on. The invisible half
was worse: the scan reads its settings in ``views/reading_view.py``, on a run
where Settings is not on screen — so **no Settings value had ever reached a
scan, in any build**. The strict-quality toggle, the TTA toggle, the ABCDE
preprocess toggle and the staff pixels-per-mm override were all inert.

The fix has two halves:

* **Canonical keys are not widget keys.** Widgets use ``ui_<name>`` and commit
  into the canonical key through an ``on_change`` callback. Streamlit only
  collects the widget key; the canonical one is an ordinary dict entry and
  survives every rerun.
* **One table, not four files.** ``reading_view`` and ``pipeline`` each used to
  re-derive a default independently, which is exactly why nobody noticed they
  were reading a value the Settings screen could not deliver. :data:`SPEC` is
  now the only place a default is written down.

Persistence is to ``data_dir()/settings.json`` because the deployment that
matters is an unattended kiosk: staff set the device up once at the start of an
event, and a browser refresh or a power cycle must not quietly undo it.

``settings.json`` holds device configuration and no participant data, so
``docs/PRIVACY.md`` is unaffected — and "End event — erase all" must **not**
delete it. That button erases scans, not the device's setup.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from services.scale import default_pixels_per_mm
from services.storage import data_dir

logger = logging.getLogger("dermascan.settings")


def _default_pi_url() -> str:
    return os.environ.get("PI_BASE_URL", "http://raspberrypi.local:5000")


def _default_keras_path() -> str:
    root = Path(__file__).resolve().parent.parent.parent
    return os.environ.get("SKIN_KERAS_PATH", str(root / "models" / "skin_classifier_full.keras"))


@dataclass(frozen=True)
class Setting:
    """One row of :data:`SPEC`.

    ``default`` is a *factory* rather than a value on purpose: the scale and the
    Pi URL are read off the environment, and this module is imported long before
    ``run_pi.sh`` has necessarily finished exporting anything.
    """

    name: str
    default: Callable[[], Any]
    env: str | None = None
    persist: bool = True
    choices: tuple[Any, ...] | None = None
    bounds: tuple[float, float] | None = None


SPEC: tuple[Setting, ...] = (
    # Off by default. settings_view.py has always *said* this, but the default
    # was once True, so an advisory ("slightly blurry") blocked the scan
    # outright — measured against real HAM10000 dermoscopy that refused 72% of
    # genuine lesions. services.lesion_gate is what stops junk input, and it
    # runs unconditionally; this switch only tightens it.
    Setting("strict_checking", lambda: False, env="SKIN_STRICT"),
    # Opt-in per session, never persisted. MVP 2 locked the design as
    # doctor-reviewed canned text by default, with the LLM reword layer a
    # deliberate per-session act — writing it to disk would quietly turn that
    # into a device default nobody re-consented to.
    Setting("assistant_gemma_enabled", lambda: False, persist=False),
    Setting(
        "inference_backend_kind",
        lambda: "local",
        choices=("local", "mock", "pi"),
    ),
    Setting("pi_base_url_input", _default_pi_url),
    Setting("SKIN_KERAS_PATH_UI", _default_keras_path),
    Setting("pixels_per_mm_ui", default_pixels_per_mm, bounds=(0.1, 100.0)),
    Setting("preprocess_enabled", lambda: True, env="SKIN_PREPROCESS"),
    Setting("preprocess_debug", lambda: False),
    # TTA on for every backend. It used to be switched off for the remote-Pi
    # backend as a speed measure, but docs/METRICS.md validated the deployed
    # 0.911 cancer sensitivity *with 4-view TTA on* — so that special case
    # quietly served a configuration nobody had measured. It is also not where
    # the time goes: the extra passes are ~0.6 s of what was a 30 s scan.
    # Staff can still turn it off; that is a deliberate, visible act.
    Setting("tta_toggle", lambda: True, env="SKIN_TTA"),
)

_BY_NAME: dict[str, Setting] = {s.name: s for s in SPEC}

# Widget keys the Settings screen is allowed to use that are NOT settings: a
# one-shot confirmation whose collection between runs is the point, not a bug.
# tests/test_settings_persist.py checks this list against the view, so a new
# widget cannot silently recreate the defect this module exists to fix.
EPHEMERAL_WIDGET_KEYS: frozenset[str] = frozenset({"reset_confirm"})


def widget_key(name: str) -> str:
    """The Streamlit ``key`` for a setting's widget. Never the canonical key."""
    return f"ui_{name}"


def _session() -> Any | None:
    """``st.session_state``, or ``None`` when there is no script run context.

    Tests, ``scripts/*`` and the Pi worker import the pipeline without ever
    starting Streamlit. This is the same ``try/except`` shape those callers
    already used inline; it lives here now so there is one copy of it.
    """
    try:
        import streamlit as st

        # Touching the mapping is what actually raises when there is no run
        # context, so do it here rather than letting a caller trip over it.
        "route" in st.session_state
        return st.session_state
    except Exception:  # noqa: BLE001 — no run context is a normal case, not an error
        return None


def _coerce(spec: Setting, value: Any) -> Any:
    """Force ``value`` into the shape of ``spec``'s default, or raise.

    Only used on values arriving from outside the process — the environment and
    the persisted file — both of which can be hand-edited on the kiosk.
    """
    default = spec.default()
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, float):
        out = float(value)
        if spec.bounds and not (spec.bounds[0] <= out <= spec.bounds[1]):
            raise ValueError(f"{spec.name}={out} outside {spec.bounds}")
        return out
    out = str(value)
    if spec.choices and out not in spec.choices:
        raise ValueError(f"{spec.name}={out!r} not in {spec.choices}")
    return out


def _from_env(spec: Setting) -> Any | None:
    if not spec.env:
        return None
    raw = os.environ.get(spec.env)
    if raw is None or raw == "":
        return None
    try:
        return _coerce(spec, raw)
    except (TypeError, ValueError):
        logger.warning("ignoring malformed %s=%r", spec.env, raw)
        return None


def get(name: str) -> Any:
    """The current value: session state, else the environment, else the default.

    **This never reads the persisted file.** If a headless read consulted
    ``~/.dermascan/settings.json`` the test suite's behaviour would depend on
    whatever a developer last toggled on a kiosk, and a scan run from a script
    would silently inherit a UI choice. The file is read exactly once, by
    :func:`init_session`, which only ever runs inside a Streamlit run.
    """
    spec = _BY_NAME[name]
    session = _session()
    if session is not None and name in session:
        return session[name]
    env = _from_env(spec)
    return spec.default() if env is None else env


def get_bool(name: str) -> bool:
    return bool(get(name))


def set_value(name: str, value: Any) -> None:
    """Write a value into session state and mirror it to disk and the env."""
    spec = _BY_NAME[name]
    session = _session()
    if session is None:
        return
    session[name] = _coerce(spec, value)
    apply_env()
    save_persisted()


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_persisted() -> dict[str, Any]:
    """Whatever the file holds, validated key by key. Never raises.

    Key by key, not wholesale: one hand-edited line should cost that one
    setting, not silently reset the whole device.
    """
    path = settings_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("ignoring unreadable %s: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("ignoring %s: expected an object", path)
        return {}
    out: dict[str, Any] = {}
    for name, value in raw.items():
        spec = _BY_NAME.get(name)
        if spec is None or not spec.persist:
            continue
        try:
            out[name] = _coerce(spec, value)
        except (TypeError, ValueError) as exc:
            logger.warning("ignoring %s in %s: %s", name, path, exc)
    return out


def save_persisted() -> None:
    """Best effort. A kiosk with a read-only home must not raise on a toggle.

    Written through a temporary file so a power cut mid-write leaves the old
    settings rather than a truncated file the next boot has to recover from.
    """
    session = _session()
    if session is None:
        return
    payload = {s.name: session[s.name] for s in SPEC if s.persist and s.name in session}
    path = settings_path()
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:  # noqa: PERF203 — one write, one place to log it
        logger.warning("could not save settings to %s: %s", path, exc)


def apply_env() -> None:
    """Push settings that the headless pipeline reads back into the environment.

    ``services.pipeline`` reads ``SKIN_TTA`` from ``os.environ`` because it also
    runs with no Streamlit at all. That write used to live in the Settings view,
    so it only happened while Settings was on screen: turn TTA off, navigate
    away, and the environment said 0 while the toggle had reverted to on. Doing
    it on every run is what keeps the two from disagreeing.

    ``os.environ`` is process-wide and therefore shared across browser sessions.
    That is a pre-existing property of how the pipeline reads TTA, not something
    introduced here, and it is harmless on the one-session kiosk.
    """
    session = _session()
    if session is None:
        return
    for spec in SPEC:
        if not spec.env or spec.name not in session:
            continue
        value = session[spec.name]
        os.environ[spec.env] = ("1" if value else "0") if isinstance(value, bool) else str(value)


def init_session() -> None:
    """Seed every canonical key: persisted file, else environment, else default.

    ``setdefault`` on a key that already exists is a no-op, which is the whole
    point — because these are not widget keys, Streamlit never collects them, so
    on the second and every later run the values are already there.
    """
    session = _session()
    if session is None:
        return
    persisted = load_persisted()
    for spec in SPEC:
        if spec.name in session:
            continue
        if spec.persist and spec.name in persisted:
            session[spec.name] = persisted[spec.name]
            continue
        env = _from_env(spec)
        session[spec.name] = spec.default() if env is None else env
    apply_env()


def commit(name: str) -> None:
    """``on_change`` callback: copy the widget's value into the canonical key.

    Streamlit runs ``on_change`` before the script body, so by the time anything
    else in the run reads the canonical key it is already correct.
    """
    session = _session()
    if session is None:
        return
    key = widget_key(name)
    if key not in session:
        return
    spec = _BY_NAME[name]
    try:
        session[name] = _coerce(spec, session[key])
    except (TypeError, ValueError) as exc:
        logger.warning("rejecting %s=%r: %s", name, session[key], exc)
        session[key] = session.get(name, spec.default())
        return
    apply_env()
    save_persisted()
