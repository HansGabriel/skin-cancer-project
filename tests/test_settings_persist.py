"""Settings must survive a screen change, a reload and a restart.

The defect this file pins was reported from real use and was invisible to every
other test: turning a Settings switch on, walking to another screen and back,
and finding it off again.

The cause is a Streamlit lifecycle detail. A keyed widget's ``session_state``
entry is garbage-collected on any run where that widget is not rendered, and
``navigation.navigate()`` makes every screen change a full rerun on which the
Settings screen is not drawn. Every setting used its canonical key as its widget
key, so every setting was collected — and ``app._init_session`` then re-seeded
the default over the top.

The half nobody could see mattered more. ``views/reading_view.py`` reads its
settings on the reading screen, which is one of those reruns — so **no Settings
value had ever reached a scan, in any build.** The strict switch, the TTA
toggle, the ABCDE preprocess toggle and the staff pixels-per-mm override were
all inert. ``test_the_scan_is_told_about_the_strict_switch`` is the test for
that half, and it is the one worth keeping if the others ever look redundant.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import streamlit as st

from services import settings

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A Streamlit run whose session state is a plain dict and whose data
    directory is disposable."""
    monkeypatch.setenv("DERMASCAN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    settings.init_session()
    return st.session_state


def _leave_the_settings_screen(session) -> None:
    """What Streamlit does on the next rerun: collect every widget key.

    Then ``app._init_session`` runs again, which is where the old code lost the
    value.
    """
    for key in [k for k in session if k.startswith("ui_")]:
        del session[key]
    settings.init_session()


def _flip(session, name: str, value) -> None:
    """Staff changing a control on the Settings screen."""
    session[settings.widget_key(name)] = value
    settings.commit(name)


def test_a_changed_setting_survives_leaving_the_screen(session):
    _flip(session, "strict_checking", True)

    _leave_the_settings_screen(session)

    assert session["strict_checking"] is True
    assert settings.get_bool("strict_checking") is True


def test_a_changed_setting_survives_a_reload(session, tmp_path, monkeypatch):
    _flip(session, "strict_checking", True)

    # A browser reload, or a restart: a brand new session state, same disk.
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    settings.init_session()

    assert st.session_state["strict_checking"] is True


def test_the_scan_is_told_about_the_strict_switch(session, monkeypatch):
    """The half of the defect that never reached the screen.

    ``reading_view`` runs on a rerun where no Settings widget exists, so this
    asserts through the view rather than through ``settings.get`` alone.
    """
    from views import reading_view

    _flip(session, "strict_checking", True)
    _leave_the_settings_screen(session)

    session["capture_image_bytes"] = b"jpeg"
    seen: dict = {}

    def _fake_scan(*args, **kwargs):
        seen.update(kwargs)
        return {"blocked": False}

    monkeypatch.setattr(reading_view, "run_scan_and_store", _fake_scan)
    monkeypatch.setattr(reading_view, "navigate", lambda *a, **k: None)
    monkeypatch.setattr(reading_view, "render_head", lambda *a, **k: None)
    monkeypatch.setattr(reading_view, "_render_checklist", lambda *a, **k: None)
    monkeypatch.setattr(reading_view.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(reading_view.st, "empty", lambda *a, **k: None)

    reading_view.render_reading_view(backend=object(), kind="local")

    assert seen["strict"] is True


def test_the_environment_agrees_with_the_toggle_after_navigating_away(session, monkeypatch):
    """``SKIN_TTA`` used to be written only while Settings was on screen.

    So turning TTA off and walking away left the environment saying 0 while the
    toggle had reverted to on — the scan and the UI disagreeing about a setting
    that changes a measured sensitivity figure.
    """
    _flip(session, "tta_toggle", False)
    _leave_the_settings_screen(session)

    import os

    assert os.environ["SKIN_TTA"] == "0"
    assert session["tta_toggle"] is False


def test_headless_reads_fall_back_to_env_then_default_and_never_to_disk(tmp_path, monkeypatch):
    """No script run context — tests, scripts/, the Pi worker.

    The disk file must be ignored here. If a headless read consulted it, the
    suite's behaviour would depend on whatever a developer last toggled on a
    kiosk, and a scan run from a script would silently inherit a UI choice.
    """
    monkeypatch.setenv("DERMASCAN_DATA_DIR", str(tmp_path))
    (tmp_path / "settings.json").write_text(json.dumps({"preprocess_enabled": False}))

    class _NoContext:
        def __contains__(self, key):
            raise RuntimeError("no script run context")

    monkeypatch.setattr(st, "session_state", _NoContext(), raising=False)

    monkeypatch.delenv("SKIN_PREPROCESS", raising=False)
    assert settings.get_bool("preprocess_enabled") is True  # the default, not the file

    monkeypatch.setenv("SKIN_PREPROCESS", "0")
    assert settings.get_bool("preprocess_enabled") is False


def test_a_corrupt_file_costs_one_setting_not_the_device(session, tmp_path):
    """Key by key, not wholesale. One bad hand-edited line should not silently
    reset a kiosk somebody spent ten minutes setting up."""
    _flip(session, "strict_checking", True)
    _flip(session, "pixels_per_mm_ui", 22.5)
    settings.settings_path().write_text(
        json.dumps({"strict_checking": True, "pixels_per_mm_ui": "not a number"})
    )

    loaded = settings.load_persisted()

    assert loaded["strict_checking"] is True
    assert "pixels_per_mm_ui" not in loaded


def test_unreadable_json_is_ignored_rather_than_raising(session):
    settings.settings_path().write_text("{ this is not json")

    assert settings.load_persisted() == {}


def test_saving_survives_an_unwritable_data_dir(session, tmp_path, monkeypatch):
    # A traceback here would replace the app on a screen with no keyboard.
    monkeypatch.setenv("DERMASCAN_DATA_DIR", str(tmp_path / "file"))
    (tmp_path / "file").write_text("not a directory")

    settings.save_persisted()  # must not raise


def test_the_assistant_reword_layer_is_never_written_to_disk(session):
    """MVP 2 locked it as opt-in *per session*. Persisting it would quietly turn
    a deliberate act into a device default nobody re-consented to."""
    _flip(session, "assistant_gemma_enabled", True)

    assert session["assistant_gemma_enabled"] is True
    assert "assistant_gemma_enabled" not in json.loads(settings.settings_path().read_text())


def test_a_rejected_value_leaves_the_setting_alone(session):
    _flip(session, "inference_backend_kind", "not-a-backend")

    assert session["inference_backend_kind"] == "local"


def test_every_settings_widget_goes_through_the_settings_module():
    """The guard that stops this defect being reinvented.

    Any widget on the Settings screen whose ``key`` is a bare name owns its own
    state, and Streamlit will collect it the moment staff navigate away. Every
    key must be ``ui_<name>`` for a SPEC entry, or an explicitly listed
    one-shot.
    """
    source = (ROOT / "frontend" / "views" / "settings_view.py").read_text()
    tree = ast.parse(source)

    keys = {
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "key" and isinstance(kw.value, ast.Constant)
    }
    allowed = settings.EPHEMERAL_WIDGET_KEYS | {
        settings.widget_key(s.name) for s in settings.SPEC
    }
    # Buttons carry keys too, and a button has no state to lose.
    buttons = {"staff_lock", "kiosk_exit", "kiosk_wipe"}

    assert keys - allowed - buttons == set()
