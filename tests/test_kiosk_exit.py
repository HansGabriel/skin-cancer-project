"""The kiosk's quit contract with launch_kiosk.sh.

The script polls for a flag file and tears the session down when it appears.
That handshake is the only way to close the kiosk from the touchscreen, so it
is worth pinning: a silent failure here strands whoever is running the demo
with no keyboard and no way out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services import kiosk


def test_request_quit_writes_the_flag(tmp_path, monkeypatch):
    flag = tmp_path / "dermascan_quit"
    monkeypatch.setattr(kiosk, "QUIT_FLAG", flag)

    kiosk.request_quit()

    assert flag.read_text() == "quit"


def test_request_quit_survives_an_unwritable_tmp(tmp_path, monkeypatch):
    # A traceback here would replace the app on the one screen that has no
    # keyboard to recover from it, so the failure has to stay silent.
    monkeypatch.setattr(kiosk, "QUIT_FLAG", tmp_path / "missing" / "quit")

    kiosk.request_quit()  # must not raise


@pytest.mark.parametrize("env,expected", [("1", True), ("0", False), (None, False)])
def test_is_kiosk_reads_the_env_contract(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("SKIN_KIOSK", raising=False)
    else:
        monkeypatch.setenv("SKIN_KIOSK", env)

    assert kiosk.is_kiosk() is expected


def test_quit_flag_path_matches_the_launcher():
    """The path is duplicated in launch_kiosk.sh; drift silently breaks Exit."""
    script = (Path(__file__).resolve().parents[1] / "launch_kiosk.sh").read_text()

    assert str(kiosk.QUIT_FLAG) in script
