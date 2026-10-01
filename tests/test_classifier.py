"""The screening rule and the calibration, as plain maths."""

from __future__ import annotations

import numpy as np
import pytest

from dermascan import config
from dermascan.classifier import apply_temperature, decide_index, load_thresholds, softmax, to_input_tensor

THRESHOLDS = load_thresholds(config.THRESHOLDS_PATH)


def test_threshold_file_is_the_one_the_rule_expects() -> None:
    assert THRESHOLDS["precancer_idx"] == 1
    assert THRESHOLDS["malignant_idx"] == 2
    assert 0 < THRESHOLDS["screen_cancer_threshold"] < 1


def _with_cancer_mass(mass: float, pre_share: float) -> list[float]:
    """[benign, pre_cancerous, malignant] with `mass` of probability on the two cancer classes."""
    return [1.0 - mass, mass * pre_share, mass * (1.0 - pre_share)]


# Built from the shipped threshold, so a retrain that moves it does not break the test.
_THR = THRESHOLDS["screen_cancer_threshold"]


@pytest.mark.parametrize(
    "probs,expected",
    [
        ([1.0, 0.0, 0.0], 0),  # plainly benign
        (_with_cancer_mass(_THR - 0.005, 0.5), 0),  # just under the line
        (_with_cancer_mass(_THR + 0.005, 0.5), 2),  # just over, flagged; tie goes to malignant
        (_with_cancer_mass(_THR + 0.02, 2 / 3), 1),  # flagged, pre_cancerous larger
        (_with_cancer_mass(_THR + 0.02, 1 / 3), 2),  # flagged, malignant larger
        ([0.05, 0.9, 0.05], 1),
        ([0.05, 0.05, 0.9], 2),
    ],
)
def test_screening_rule(probs, expected) -> None:
    assert decide_index(np.asarray(probs, np.float32), THRESHOLDS) == expected


def test_without_a_threshold_file_the_rule_is_argmax() -> None:
    for probs in ([0.2, 0.5, 0.3], [0.34, 0.33, 0.33], [0.05, 0.05, 0.9]):
        p = np.asarray(probs, np.float32)
        assert decide_index(p, {}) == int(np.argmax(p))


def test_temperature_matches_logit_scaling() -> None:
    rng = np.random.default_rng(42)
    for _ in range(100):
        logits = rng.normal(0, 3, size=3)
        t = float(rng.uniform(0.5, 3.0))
        np.testing.assert_allclose(apply_temperature(softmax(logits), t), softmax(logits / t), atol=1e-5)


def test_temperature_softens_but_never_reorders() -> None:
    probs = np.array([0.05, 0.05, 0.90], np.float32)
    cal = apply_temperature(probs, 1.537)
    assert cal[2] < probs[2]
    assert np.argmax(cal) == 2
    assert cal.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(apply_temperature(np.array([0.0, 0.0, 1.0], np.float32), 1.537)).all()


def test_calibration_can_never_change_the_label() -> None:
    """The label is decided on raw probabilities; the calibrated ones are display only."""
    rng = np.random.default_rng(7)
    for _ in range(1000):
        p = rng.dirichlet(np.ones(3)).astype(np.float32)
        assert int(np.argmax(apply_temperature(p, 1.537))) == int(np.argmax(p))
        assert decide_index(p, THRESHOLDS) == decide_index(p, THRESHOLDS)


def test_input_tensor_is_float_0_255_unless_the_model_is_quantised() -> None:
    rgb = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    f = to_input_tensor(rgb, {"dtype": np.float32, "quantization": (0.0, 0)})
    assert f.shape == (1, 224, 224, 3) and f.dtype == np.float32 and f.max() > 1.0
    q = to_input_tensor(rgb, {"dtype": np.int8, "quantization": (1.0, -128)})
    assert q.dtype == np.int8 and int(q.min()) == int(rgb.min()) - 128


def test_real_model_answers_with_four_views(classifier) -> None:
    rgb = np.full((300, 300, 3), (196, 158, 132), np.uint8)
    p = classifier.predict(rgb)
    assert p.label in classifier.labels
    assert set(p.probs_pct) == set(classifier.labels)
    assert abs(sum(p.raw_probs) - 1.0) < 1e-3
    assert 0 <= p.confidence_pct <= 100
    assert p.flagged == (p.label != "benign")
