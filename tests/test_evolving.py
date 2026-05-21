from __future__ import annotations

import json

import numpy as np
import pytest

from services.evolving import compute, to_letter_result
from services.storage import Storage


def _abcd(d_mm: float = 5.0) -> dict:
    base = {"value": 0.1, "tier": 0, "verdict": "normal"}
    return {
        "A": base,
        "B": base,
        "C": {**base, "value": 2},
        "D": {**base, "value": d_mm},
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMASCAN_DATA_DIR", str(tmp_path))
    import services.storage as sm

    sm._storage = None
    return Storage(tmp_path)


def _rgb_mask() -> tuple[np.ndarray, np.ndarray]:
    rgb = np.full((80, 80, 3), 120, dtype=np.uint8)
    rgb[25:55, 25:55] = 200
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[25:55, 25:55] = 255
    return rgb, mask


def test_needs_history_without_prior_scan(store: Storage) -> None:
    ev = compute("no-case", _abcd())
    assert ev.verdict == "needs history"
    assert to_letter_result(ev)["detail"] == "NEEDS HISTORY"


def test_second_save_gets_evolving_verdict(store: Storage) -> None:
    import cv2

    f = store.create_folder("F")
    c = store.create_case(f.id, "Test", body_site="arm")
    rgb, mask = _rgb_mask()
    pl1 = {
        "composite": 20.0,
        "risk_band": "low",
        "quality": {"ok": True, "reasons": [], "reason_details": []},
        "abcde": {**_abcd(4.0), "E": {"value": None, "tier": 0, "verdict": "needs history"}},
        "scan_result": type("SR", (), {"label": "benign", "confidence": 80.0, "probs": {"malignant": 5.0}})(),
        "pixels_per_mm": 10.0,
        "rgb": rgb,
        "mask": mask,
    }
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    jpg = buf.tobytes()
    store.save_scan(c.id, pl1, jpg)
    pl2 = dict(pl1)
    pl2["abcde"] = {**_abcd(6.5), "E": {"value": None, "tier": 0, "verdict": "needs history"}}
    s2 = store.save_scan(c.id, pl2, jpg)
    e = json.loads(s2.e_json or "{}")
    assert e.get("verdict") in ("watch", "changing", "stable")
    assert e.get("verdict") != "needs history"
