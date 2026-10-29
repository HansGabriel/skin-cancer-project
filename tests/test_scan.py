"""End to end through run_scan with the real model file."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from conftest import NEUTRAL, SKIN_TONES, jpeg, textured, with_lesion
from dermascan import config, scan


def test_a_lesion_gets_a_verdict_quickly(classifier) -> None:
    t = time.perf_counter()
    out = scan.run_scan(jpeg(with_lesion(textured(SKIN_TONES["light"], size=1024))), classifier=classifier)
    assert time.perf_counter() - t < 2.0
    assert out.status == "ok"
    assert out.prediction is not None and out.verdict.state in ("low_concern", "uncertain", "urgent", "needs_attention", "uncertain_caution")
    assert set(out.stage_ms) == {"decode", "gate", "model", "signs"}
    assert out.to_dict()["total_ms"] == sum(out.stage_ms.values())
    assert [s["letter"] for s in out.to_dict()["signs"]] == ["A", "B", "C", "D", "E"]
    assert [line["letter"] for line in out.to_dict()["sign_lines"]] == ["A", "B", "C"]


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
    # No spot in the photo: the model's numbers are for staff, the visitor gets no verdict.
    assert forced.verdict.state == "cannot_judge" and forced.prediction is not None
    d = forced.to_dict()
    assert d["judged"] is False and d["caveat"] == "" and d["prediction"]["cancer_score"] is not None


def test_a_forced_read_of_a_badly_framed_spot_keeps_its_verdict(classifier, monkeypatch) -> None:
    """Framing refusals have a real spot in the shot, so "read anyway" still answers, with the caveat."""
    from dermascan import gate

    real_check = gate.check

    def off_centre(rgb):
        import dataclasses
        return dataclasses.replace(real_check(rgb), refusal=gate.Refusal("not_in_ring", True))

    monkeypatch.setattr(gate, "check", off_centre)
    out = scan.run_scan(jpeg(with_lesion(textured(SKIN_TONES["light"]))), force=True, classifier=classifier)
    assert out.status == "ok" and out.forced and out.verdict.state != "cannot_judge"
    assert out.to_dict()["judged"] is True and out.to_dict()["caveat"]
    # The outline is a real mark here, so A B C are measured and shown (since 2026-10-29).
    assert [s["letter"] for s in out.to_dict()["signs"]] == ["A", "B", "C", "D", "E"]


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
    forced = scan.run_scan(photo, force=True, classifier=classifier).to_dict()
    assert forced["caveat"] == "" and forced["verdict"]["headline"] == "WE CAN'T JUDGE THIS PHOTO"  # the headline says it
    ok = scan.run_scan(jpeg(with_lesion(textured(SKIN_TONES["light"]))), classifier=classifier)
    assert ok.to_dict()["caveat"] == ""


def test_two_threads_sharing_one_classifier_get_their_own_answers(classifier) -> None:
    """The web demo serves each visitor on its own thread with one shared model."""
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


def test_a_checked_photo_is_not_decoded_or_gated_twice(classifier) -> None:
    from dermascan.scan import check_photo, check_result, run_check, run_scan

    data = jpeg(with_lesion(textured(SKIN_TONES["light"])))
    checked = check_photo(data)
    assert checked.rgb is not None and checked.report is not None and checked.report.small is not None
    assert check_result(checked) == run_check(data)
    out = run_scan(data, classifier=classifier, checked=checked)
    assert out.status == "ok" and out.stage_ms["decode"] == 0 and out.stage_ms["gate"] == 0
    assert out.measured == checked.report.measured and out.abc_signs
    assert run_scan(data, classifier=classifier).prediction.label == out.prediction.label
    bad = check_photo(b"not a jpeg")
    assert bad.rgb is None and check_result(bad)["status"] == "error"
    assert run_scan(b"not a jpeg", classifier=classifier, checked=bad).status == "error"


def test_a_picture_too_large_to_open_is_not_a_picture() -> None:
    """A small PNG that claims a huge picture is refused, not decoded into gigabytes.

    OpenCV reads its limit once, when cv2 is first imported, so this runs in a fresh
    process that imports dermascan first, as the kiosk and the web demo do. A limit
    set from outside must not loosen it.
    """
    code = (
        "import dermascan, cv2, numpy as np\n"
        "from dermascan.scan import check_photo\n"
        "big = cv2.imencode('.png', np.zeros((8000, 8000, 3), np.uint8))[1].tobytes()  # 64 MP, ~200 KB\n"
        "assert check_photo(big).rgb is None\n"
        "phone = cv2.imencode('.jpg', np.zeros((3000, 4000, 3), np.uint8))[1].tobytes()  # 12 MP\n"
        "assert check_photo(phone).rgb is not None\n"
    )
    env = {**os.environ, "OPENCV_IO_MAX_IMAGE_PIXELS": "999999999"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
