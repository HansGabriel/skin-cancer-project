"""Decision/calibration parity: scripts/pi_server.py must mirror backend.tflite_shared.

The Pi Flask server carries its own copy of the sensitivity-first decision logic
and the temperature calibration (it must stay dependency-light on the Pi, so it
cannot import backend/). These tests feed the same probability vectors through
both implementations and require identical answers. If backend/tflite_shared.py
changes, scripts/pi_server.py must change with it — this suite is the tripwire.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import tflite_shared  # noqa: E402
from scripts import pi_server  # noqa: E402

# Same source of truth for both sides: backend's loaded thresholds dict is passed
# verbatim to pi_server's parametrised decide_index.
THRESHOLDS = tflite_shared._load_thresholds()

# Probability vectors covering the interesting regions of the simplex
# (labels order: benign, pre_cancerous, malignant; screen threshold 0.11 on
# p(pre_cancerous) + p(malignant)).
GRID = [
    [0.34, 0.33, 0.33],            # near-uniform
    [1.0, 0.0, 0.0],               # one-hot benign
    [0.0, 1.0, 0.0],               # one-hot pre_cancerous
    [0.0, 0.0, 1.0],               # one-hot malignant
    [0.89, 0.055, 0.055],          # p_pre+p_mal == 0.11 exactly, pre/mal tied
    [0.895, 0.0525, 0.0525],       # just below the screen threshold (0.105)
    [0.885, 0.0575, 0.0575],       # just above the screen threshold (0.115)
    [0.89, 0.11, 0.0],             # threshold cleared by pre_cancerous alone
    [0.89, 0.0, 0.11],             # threshold cleared by malignant alone
    [0.90, 0.06, 0.04],            # 0.10 cancer mass — below threshold
    [0.88, 0.04, 0.08],            # 0.12 cancer mass — flagged, malignant wins
    [0.88, 0.08, 0.04],            # 0.12 cancer mass — flagged, pre wins
    [0.5, 0.25, 0.25],
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9],
    [0.333333, 0.333333, 0.333334],
]

TEMPERATURES = (1.0, 1.5369100196927983, 2.0, 0.5)


def _grid() -> list[np.ndarray]:
    vectors = [np.asarray(v, dtype=np.float32) for v in GRID]
    rng = np.random.default_rng(20260808)
    vectors += [rng.dirichlet(np.ones(3)).astype(np.float32) for _ in range(500)]
    return vectors


def test_threshold_config_present() -> None:
    """The parity grid is designed around the deployed screen threshold."""
    assert "screen_cancer_threshold" in THRESHOLDS
    assert THRESHOLDS["precancer_idx"] == 1
    assert THRESHOLDS["malignant_idx"] == 2


@pytest.mark.parametrize("probs", _grid(), ids=lambda p: np.array2string(p, precision=4))
def test_decide_index_parity(probs: np.ndarray) -> None:
    backend_idx = tflite_shared.decide_index(probs)
    pi_idx = pi_server.decide_index(probs, THRESHOLDS)
    assert pi_idx == backend_idx


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_apply_temperature_parity(temperature: float) -> None:
    for probs in _grid():
        backend_cal = tflite_shared.apply_temperature(probs, temperature)
        pi_cal = pi_server.apply_temperature(probs, temperature)
        np.testing.assert_allclose(pi_cal, backend_cal, atol=1e-6)
        assert int(np.argmax(pi_cal)) == int(np.argmax(backend_cal))


def test_displayed_label_and_confidence_parity() -> None:
    """End-to-end: same decided label AND same displayed confidence both sides."""
    T = 1.5369100196927983
    for probs in _grid():
        idx_backend = tflite_shared.decide_index(probs)
        idx_pi = pi_server.decide_index(probs, THRESHOLDS)
        assert idx_pi == idx_backend
        conf_backend = float(tflite_shared.apply_temperature(probs, T)[idx_backend])
        conf_pi = float(pi_server.apply_temperature(probs, T)[idx_pi])
        assert conf_pi == pytest.approx(conf_backend, abs=1e-6)


def test_no_threshold_config_falls_back_to_argmax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tflite_shared, "_THRESHOLDS", {})
    for probs in ([0.2, 0.5, 0.3], [0.34, 0.33, 0.33], [0.05, 0.05, 0.9]):
        p = np.asarray(probs, dtype=np.float32)
        assert tflite_shared.decide_index(p) == int(np.argmax(p))
        assert pi_server.decide_index(p, {}) == int(np.argmax(p))
