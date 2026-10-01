"""training/calibrate.py and training/eval.py, on a tiny fake split, with the real model.

These run on the PC after a retrain; a crash there costs a training day, so they are
exercised here on every test run instead.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, SKIN_TONES, jpeg, textured, with_lesion

sys.path.insert(0, str(ROOT / "training"))
import calibrate  # noqa: E402


def test_threshold_is_the_highest_that_keeps_the_target_sensitivity() -> None:
    # 4 cancers with cancer-probability 0.05, 0.10, 0.30, 0.80; 4 benign with 0.01-0.20.
    truth = np.array([2, 2, 1, 2, 0, 0, 0, 0])
    p_cancer = np.array([0.05, 0.10, 0.30, 0.80, 0.01, 0.02, 0.15, 0.20])
    probs = np.stack([1 - p_cancer, p_cancer / 2, p_cancer / 2], axis=1)
    out = calibrate.fit_threshold(truth, probs, [1, 2], target=0.75)
    assert out["screen_cancer_threshold"] == pytest.approx(0.10)  # 3 of 4 cancers still flagged
    assert out["val_sensitivity"] == pytest.approx(0.75)
    assert out["val_specificity"] == pytest.approx(0.5)


def test_temperature_softens_an_overconfident_model() -> None:
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, 400)
    # Right 70% of the time but always 98% sure: should need T > 1.
    guess = np.where(rng.random(400) < 0.7, truth, (truth + 1) % 3)
    probs = np.full((400, 3), 0.01)
    probs[np.arange(400), guess] = 0.98
    out = calibrate.fit_temperature(truth, probs)
    assert out["T"] > 1.0 and out["val_nll_after"] < out["val_nll_before"]


@pytest.fixture()
def fake_split(tmp_path: Path) -> Path:
    rows = []
    for i in range(12):
        img = with_lesion(textured(SKIN_TONES["light"], size=300, seed=i)) if i % 2 else textured(SKIN_TONES["medium"], size=300, seed=i)
        path = tmp_path / f"ISIC_{i:07d}.jpg"
        path.write_bytes(jpeg(img))
        split = "val" if i < 6 else "test"
        rows.append({"split": split, "lesion_id": f"L{i}", "image_id": path.stem, "dx": "nv", "label_3": "benign", "label_idx": i % 3, "path": str(path)})
    csv_path = tmp_path / "splits.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return csv_path


def test_eval_runs_on_a_split_and_prints_the_metrics_table(fake_split: Path) -> None:
    out = subprocess.run([sys.executable, "training/eval.py", "--csv", str(fake_split)], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Scoring 6 test images" in out.stdout
    assert "| Cancer sensitivity (at threshold) |" in out.stdout


def test_calibrate_writes_both_files(fake_split: Path, tmp_path: Path, monkeypatch) -> None:
    from dermascan import config

    monkeypatch.setattr(config, "THRESHOLDS_PATH", tmp_path / "thresholds.json")
    monkeypatch.setattr(config, "TEMPERATURE_PATH", tmp_path / "temperature.json")
    monkeypatch.setattr(sys, "argv", ["calibrate.py", "--csv", str(fake_split), "--target", "0.5"])
    assert calibrate.main() == 0
    thr = json.loads((tmp_path / "thresholds.json").read_text())
    assert thr["precancer_idx"] == 1 and thr["malignant_idx"] == 2
    assert 0 < thr["screen_cancer_threshold"] < 1 and thr["val_sensitivity"] >= 0.5
    assert json.loads((tmp_path / "temperature.json").read_text())["T"] > 0


def test_missing_split_file_is_a_sentence_not_a_traceback(tmp_path: Path) -> None:
    out = subprocess.run([sys.executable, "training/eval.py", "--csv", str(tmp_path / "nope.csv")], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode != 0
    assert "Run the training notebook first" in out.stderr and "Traceback" not in out.stderr
