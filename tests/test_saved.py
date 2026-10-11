"""kiosk/saved.py on its own: what reaches the folder, and what comes back after a restart."""

from __future__ import annotations

import json

import pytest

from conftest import SKIN_TONES, jpeg, textured, with_lesion
from dermascan import config
from kiosk.saved import SavedScans

PHOTO = jpeg(with_lesion(textured(SKIN_TONES["light"])))
OUTCOME = {"status": "ok", "verdict": {"headline": "NOTHING STOOD OUT", "tone": "calm", "state": "clear"}}


def test_a_save_is_two_files_and_survives_a_restart(tmp_path) -> None:
    store = SavedScans(tmp_path)
    a = store.add("Back", PHOTO, OUTCOME)
    b = store.add("Back", PHOTO, OUTCOME, group="Back")
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{a.id}.jpg", f"{a.id}.json", f"{b.id}.jpg", f"{b.id}.json"]
    assert json.loads((tmp_path / f"{a.id}.json").read_text())["site"] == "Back"

    again = SavedScans(tmp_path)  # the kiosk after a reboot
    assert [s.id for s in (again.get(a.id), again.get(b.id))] == [a.id, b.id]
    assert again.get(a.id).jpeg == a.jpeg and again.get(a.id).outcome == OUTCOME
    assert again.history(b.id)["count"] == 2 and again.history(b.id)["first"]["id"] == a.id
    c = again.add("Back", PHOTO, OUTCOME)
    assert c.id == b.id + 1  # ids continue; a new scan never takes an old one's number


def test_delete_and_erase_remove_the_files(tmp_path) -> None:
    store = SavedScans(tmp_path)
    a = store.add("Face", PHOTO, OUTCOME)
    b = store.add("Neck", PHOTO, OUTCOME)
    assert store.delete(a.id) and not (tmp_path / f"{a.id}.jpg").exists() and not (tmp_path / f"{a.id}.json").exists()
    assert (tmp_path / f"{b.id}.json").exists()
    store.erase()
    assert store.stats()["count"] == 0 and not list(tmp_path.iterdir())
    assert SavedScans(tmp_path).stats()["count"] == 0


def test_the_cap_also_removes_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "SAVED_MAX", 2)
    store = SavedScans(tmp_path)
    ids = [store.add("Back", PHOTO, OUTCOME).id for _ in range(3)]
    assert store.get(ids[0]) is None and not (tmp_path / f"{ids[0]}.json").exists()
    assert len(list(tmp_path.glob("*.json"))) == 2
    # Too many on the card from an earlier run: the oldest go at load time too.
    monkeypatch.setattr(config, "SAVED_MAX", 1)
    assert SavedScans(tmp_path).stats()["count"] == 1 and len(list(tmp_path.glob("*.json"))) == 1


def test_leftovers_from_a_power_cut_are_cleaned_up(tmp_path) -> None:
    store = SavedScans(tmp_path)
    good = store.add("Back", PHOTO, OUTCOME)
    (tmp_path / "7.jpg").write_bytes(PHOTO)  # photo written, JSON never made it
    (tmp_path / "8.json").write_text("{not json")  # half-written record
    (tmp_path / "8.jpg").write_bytes(PHOTO)
    (tmp_path / "9.json.tmp").write_text("x")
    again = SavedScans(tmp_path)
    assert [s["id"] for g in again.spots() for s in g["scans"]] == [good.id]
    assert not (tmp_path / "7.jpg").exists() and not (tmp_path / "9.json.tmp").exists()
    again.erase()  # End event removes the unreadable record as well
    assert not list(tmp_path.iterdir())


def test_memory_only_when_no_folder(tmp_path) -> None:
    store = SavedScans(None)
    store.add("Back", PHOTO, OUTCOME)
    assert store.stats()["count"] == 1 and store.stats()["folder"] is None
    assert not list(tmp_path.iterdir())


def test_a_card_that_cannot_be_written_does_not_refuse_the_save(tmp_path) -> None:
    blocker = tmp_path / "saved"
    blocker.write_text("a file where the folder should be")  # mkdir will fail
    store = SavedScans(blocker)
    scan = store.add("Back", PHOTO, OUTCOME)
    assert store.get(scan.id) is not None  # kept for this run
    with pytest.raises(ValueError):
        store.add("Nowhere", PHOTO, OUTCOME)
