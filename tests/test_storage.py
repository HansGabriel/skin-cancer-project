from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from services.storage import Storage


@dataclass
class _SR:
    label: str = "benign"
    confidence: float = 90.0
    probs: dict | None = None
    image_jpg_bytes: bytes = b"\xff\xd8\xff\xd9"

    def __post_init__(self):
        self.probs = self.probs or {"benign": 90.0, "pre_cancerous": 5.0, "malignant": 5.0}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Storage:
    monkeypatch.setenv("DERMASCAN_DATA_DIR", str(tmp_path))
    import services.storage as sm

    sm._storage = None
    return Storage(tmp_path)


def _pipeline(sr: _SR) -> dict:
    return {
        "composite": 30.0,
        "risk_band": "low",
        "quality": {"ok": True, "reasons": [], "reason_details": []},
        "abcde": {
            "A": {"value": 0.1, "tier": 0, "verdict": "normal"},
            "B": {"value": 1.0, "tier": 0, "verdict": "normal"},
            "C": {"value": 2, "tier": 0, "verdict": "normal"},
            "D": {"value": 4.0, "tier": 0, "verdict": "normal"},
            "E": {"value": None, "tier": 0, "verdict": "needs history"},
        },
        "scan_result": sr,
        "pixels_per_mm": 10.0,
    }


def test_folder_case_scan(store: Storage) -> None:
    f = store.create_folder("F")
    c = store.create_case(f.id, "C", body_site="arm")
    s = store.save_scan(c.id, _pipeline(_SR()), b"\xff\xd8\xff" + b"\x00" * 20)
    assert len(store.list_scans(c.id)) == 1
    assert json.loads(s.abcd_json)["D"]["value"] == 4.0


def test_delete_folder(store: Storage) -> None:
    f = store.create_folder("X")
    c = store.create_case(f.id, "Y")
    store.save_scan(c.id, _pipeline(_SR()), b"\xff\xd8\xff")
    store.delete_folder(f.id)
    assert store.list_folders() == []
