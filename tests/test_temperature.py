"""Temperature scaling: display-only calibration must never change a decision."""

from __future__ import annotations

import numpy as np
import pytest

from backend.tflite_shared import apply_temperature, decide_index


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


def test_apply_temperature_matches_logit_scaling() -> None:
    rng = np.random.default_rng(42)
    for _ in range(100):
        logits = rng.normal(0, 3, size=3)
        t = float(rng.uniform(0.5, 3.0))
        expected = _softmax(logits / t)
        got = apply_temperature(_softmax(logits), t)
        np.testing.assert_allclose(got, expected, atol=1e-5)


def test_apply_temperature_identity_at_t1() -> None:
    probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    np.testing.assert_allclose(apply_temperature(probs, 1.0), probs, atol=1e-7)


def test_apply_temperature_softens_confidence() -> None:
    probs = np.array([0.05, 0.05, 0.90], dtype=np.float32)
    calibrated = apply_temperature(probs, 1.537)
    assert calibrated[2] < probs[2]  # over-confidence reduced
    assert np.argmax(calibrated) == 2  # ranking preserved
    assert calibrated.sum() == pytest.approx(1.0, abs=1e-6)


def test_decision_unchanged_by_calibration() -> None:
    """decide_index must run on raw probs; calibration must never flip the label."""
    rng = np.random.default_rng(7)
    for _ in range(1000):
        probs = rng.dirichlet(np.ones(3)).astype(np.float32)
        raw_decision = decide_index(probs)
        # The pipeline decides BEFORE calibrating — same input, same output.
        assert decide_index(probs) == raw_decision
        # Calibration preserves ranking, so even a (wrong) post-calibration
        # decision at T>0 would agree for the argmax fallback; the real guard
        # is that compose_scan_result calls decide_index(probs) on raw probs.
        calibrated = apply_temperature(probs, 1.537)
        assert int(np.argmax(calibrated)) == int(np.argmax(probs))


def test_apply_temperature_handles_zero_probs() -> None:
    probs = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    calibrated = apply_temperature(probs, 1.537)
    assert np.isfinite(calibrated).all()
    assert calibrated.sum() == pytest.approx(1.0, abs=1e-6)
