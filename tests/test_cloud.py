"""The Streamlit Cloud page, driven headless. Skipped where Streamlit is not installed."""

from __future__ import annotations

import pytest

from conftest import ROOT, SKIN_TONES, jpeg, textured, with_lesion

testing = pytest.importorskip("streamlit.testing.v1")

APP = str(ROOT / "cloud" / "streamlit_app.py")


class _Upload:
    """Stands in for st.file_uploader's return value."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _run_with(monkeypatch, at, data: bytes | None):
    import streamlit as st

    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: None if data is None else _Upload(data))
    return at.run(timeout=30)


def test_empty_page_invites_an_upload(monkeypatch) -> None:
    at = _run_with(monkeypatch, testing.AppTest.from_file(APP), None)
    assert not at.exception
    assert any("Upload a photo" in m.value for m in at.markdown)


def test_a_lesion_gets_a_verdict(monkeypatch) -> None:
    at = _run_with(monkeypatch, testing.AppTest.from_file(APP), jpeg(with_lesion(textured(SKIN_TONES["light"]))))
    assert not at.exception
    assert at.subheader[0].value in ("NOTHING STOOD OUT", "NOT SURE", "SEE A DOCTOR SOON", "GET THIS CHECKED", "BETTER TO GET CHECKED")


def test_check_it_anyway_applies_to_that_photo_only(monkeypatch) -> None:
    at = testing.AppTest.from_file(APP)
    bare = jpeg(textured(SKIN_TONES["medium"]))
    _run_with(monkeypatch, at, bare)
    assert at.button[0].label == "Check it anyway"
    at.button[0].click()
    at.run(timeout=30)
    assert any("read anyway" in w.value for w in at.warning), "the forced result must carry the caveat"

    other = jpeg(textured(SKIN_TONES["light"], seed=3))  # a different photo of bare skin
    _run_with(monkeypatch, at, other)
    assert not any("read anyway" in w.value for w in at.warning), "the override leaked onto the next photo"
    assert at.button and at.button[0].label == "Check it anyway"
