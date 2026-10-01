"""End to end through run_scan with the real model file."""

from __future__ import annotations

import time

import numpy as np

from conftest import NEUTRAL, SKIN_TONES, jpeg, textured, with_lesion
from dermascan import config, scan


def test_a_lesion_gets_a_verdict_quickly(classifier) -> None:
    t = time.perf_counter()
    out = scan.run_scan(jpeg(with_lesion(textured(SKIN_TONES["light"], size=1024))), classifier=classifier)
    assert time.perf_counter() - t < 2.0
    assert out.status == "ok"
    assert out.prediction is not None and out.verdict.state in ("low_concern", "uncertain", "urgent", "needs_attention", "uncertain_caution")
    assert set(out.stage_ms) == {"decode", "gate", "model"}
    assert out.to_dict()["total_ms"] == sum(out.stage_ms.values())


def test_a_wall_never_reaches_the_model(classifier, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(classifier, "predict", lambda rgb: calls.append(1))
    out = scan.run_scan(jpeg(textured(NEUTRAL["grey desk"])), classifier=classifier)
    assert out.status == "refused" and out.refusal.code == "no_skin"
    assert not calls
    assert out.to_dict()["refusal"]["can_override"] is False


def test_bare_skin_is_refused_but_can_be_forced(classifier) -> None:
    photo = jpeg(textured(SKIN_TONES["medium"]))
    refused = scan.run_scan(photo, classifier=classifier)
    assert refused.status == "refused" and refused.refusal.can_override
    forced = scan.run_scan(photo, force=True, classifier=classifier)
    assert forced.status == "ok" and forced.forced is True
    assert forced.refusal is not None  # the screen can still say the checks were skipped


def test_a_hard_refusal_cannot_be_forced(classifier) -> None:
    out = scan.run_scan(jpeg(textured(NEUTRAL["white wall"])), force=True, classifier=classifier)
    assert out.status == "refused"


def test_garbage_bytes_are_an_error_not_a_crash(classifier) -> None:
    out = scan.run_scan(b"not a jpeg", classifier=classifier)
    assert out.status == "error"
    assert out.verdict.headline == "THAT SCAN DID NOT FINISH"
    assert "picture" in out.verdict.body


def test_a_model_failure_reaches_the_screen_without_the_exception_text(classifier, monkeypatch) -> None:
    def boom(rgb):
        raise RuntimeError("tflite internal: tensor 17 bad shape")

    monkeypatch.setattr(classifier, "predict", boom)
    out = scan.run_scan(jpeg(with_lesion(textured(SKIN_TONES["light"]))), classifier=classifier)
    assert out.status == "error"
    assert "tensor" not in out.verdict.text() and "tflite" not in out.verdict.text().lower()


def test_a_forced_result_carries_the_caveat_and_a_normal_one_does_not(classifier) -> None:
    photo = jpeg(textured(SKIN_TONES["medium"]))
    assert scan.run_scan(photo, force=True, classifier=classifier).to_dict()["caveat"]
    ok = scan.run_scan(jpeg(with_lesion(textured(SKIN_TONES["light"]))), classifier=classifier)
    assert ok.to_dict()["caveat"] == ""


def test_two_threads_sharing_one_classifier_get_their_own_answers(classifier) -> None:
    """Streamlit Cloud serves each visitor on its own thread with one shared model."""
    from concurrent.futures import ThreadPoolExecutor

    lesion = with_lesion(textured(SKIN_TONES["light"]))
    bare = textured(SKIN_TONES["deep"])
    expected = {id(lesion): classifier.predict(lesion).raw_probs, id(bare): classifier.predict(bare).raw_probs}
    with ThreadPoolExecutor(8) as pool:
        jobs = [(img, pool.submit(classifier.predict, img)) for img in [lesion, bare] * 20]
        for img, job in jobs:
            np.testing.assert_allclose(job.result().raw_probs, expected[id(img)], atol=1e-6)


def test_huge_uploads_are_reduced_before_anything_runs(classifier) -> None:
    big = with_lesion(textured(SKIN_TONES["light"], size=3000))
    out = scan.run_scan(jpeg(big), classifier=classifier)
    assert out.status == "ok"
    assert out.stage_ms["gate"] < 500


def test_captures_are_saved_only_when_asked(classifier, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "SAVE_CAPTURES_DIR", None)
    scan.run_scan(jpeg(textured(SKIN_TONES["light"])), classifier=classifier)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(config, "SAVE_CAPTURES_DIR", tmp_path)
    scan.run_scan(jpeg(textured(SKIN_TONES["light"])), classifier=classifier)
    assert len(list(tmp_path.glob("*.jpg"))) == 1 and len(list(tmp_path.glob("*.txt"))) == 1
