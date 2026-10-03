"""The Flask routes, with the camera absent (as on a laptop) and faked (as on the Pi)."""

from __future__ import annotations

import io

import pytest

from conftest import NEUTRAL, SKIN_TONES, jpeg, textured, with_lesion
from kiosk import camera, server


@pytest.fixture()
def client(monkeypatch):
    server.last_photo.forget()
    server.saved.erase()
    monkeypatch.setitem(server.settings, "allow_read_anyway", True)
    monkeypatch.setitem(server.settings, "show_staff_details", True)
    monkeypatch.delenv("DERMASCAN_PASSCODE", raising=False)
    monkeypatch.setattr(server.config, "staff_passcode", lambda: None)
    return server.app.test_client()


def test_health_reports_model_and_camera(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] and body["model"]["labels"] == ["benign", "pre_cancerous", "malignant"]
    assert body["camera"] is False


def test_page_is_served(client) -> None:
    r = client.get("/")
    assert r.status_code == 200 and b"<html" in r.data.lower()


def test_scan_without_a_photo_is_a_clear_error(client) -> None:
    r = client.post("/scan")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error" and body["verdict"]["headline"] == "THAT SCAN DID NOT FINISH"


def test_upload_scan_round_trip(client) -> None:
    data = {"image": (io.BytesIO(jpeg(with_lesion(textured(SKIN_TONES["light"])))), "spot.jpg")}
    r = client.post("/scan", data=data, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert body["verdict"]["headline"] and body["prediction"]["label"]


def test_refused_then_forced_uses_the_remembered_photo(client) -> None:
    data = {"image": (io.BytesIO(jpeg(textured(SKIN_TONES["medium"]))), "skin.jpg")}
    first = client.post("/scan", data=data, content_type="multipart/form-data").get_json()
    assert first["status"] == "refused" and first["refusal"]["can_override"]
    second = client.post("/scan?force=1").get_json()
    assert second["status"] == "ok" and second["forced"] is True
    assert second["signs"] is None and second["sign_lines"] == []  # never measured on an outline the gate rejected


def test_no_camera_routes_say_so(client) -> None:
    r = client.post("/capture")
    assert r.status_code == 404 and r.get_json()["verdict"]["body"] == "The camera did not take the photo."
    assert client.get("/preview.mjpg").status_code == 404


def test_done_forgets_the_photo(client) -> None:
    data = {"image": (io.BytesIO(jpeg(with_lesion(textured(SKIN_TONES["light"])))), "spot.jpg")}
    assert client.post("/scan", data=data, content_type="multipart/form-data").status_code == 200
    assert client.post("/forget").get_json()["ok"]
    assert client.post("/scan?force=1").status_code == 400


def test_a_scan_reads_its_own_upload_even_if_another_arrives_meanwhile(client, monkeypatch) -> None:
    """The scan must use the bytes it was sent, not whatever was remembered last."""
    mine = jpeg(with_lesion(textured(SKIN_TONES["light"])))
    seen = []

    def spy(data, force=False):
        server.last_photo.put(b"someone else's photo")  # a second request lands mid-scan
        seen.append(data)
        from dermascan.scan import run_scan
        return run_scan(data, force=force)

    monkeypatch.setattr(server, "run_scan", spy)
    client.post("/scan", data={"image": (io.BytesIO(mine), "a.jpg")}, content_type="multipart/form-data")
    assert seen == [mine]


def test_capture_and_scan_with_a_fake_camera(client, monkeypatch) -> None:
    photo = jpeg(with_lesion(textured(SKIN_TONES["light"], size=1024)))
    monkeypatch.setattr(camera.camera, "_cam", object())
    monkeypatch.setattr(camera.camera, "capture_jpeg", lambda: photo)
    r = client.post("/capture")
    assert r.status_code == 200 and r.mimetype == "image/jpeg" and r.data == photo
    assert client.post("/scan").get_json()["status"] == "ok"
    monkeypatch.setattr(camera.camera, "_cam", None)


def test_quit_writes_the_flag(client, tmp_path, monkeypatch) -> None:
    flag = tmp_path / "quit"
    monkeypatch.setattr(server.config, "QUIT_FLAG", flag)
    assert client.post("/quit").get_json()["ok"]
    assert flag.read_text() == "quit"


def test_quit_needs_the_staff_code_when_one_is_set(client, tmp_path, monkeypatch) -> None:
    flag = tmp_path / "quit"
    monkeypatch.setattr(server.config, "QUIT_FLAG", flag)
    monkeypatch.setattr(server.config, "staff_passcode", lambda: "4321")
    assert client.get("/health").get_json()["exit_needs_code"] is True
    assert client.post("/quit", data={"code": "1111"}).status_code == 403
    assert not flag.exists()
    assert client.post("/quit", data={"code": "4321"}).status_code == 200
    assert flag.exists()


def test_passcode_comes_from_env_or_the_home_file(tmp_path, monkeypatch) -> None:
    from dermascan import config

    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("DERMASCAN_PASSCODE", raising=False)
    assert config.staff_passcode() is None
    (tmp_path / ".dermascan_passcode").write_text("2468\n")
    assert config.staff_passcode() == "2468"
    monkeypatch.setenv("DERMASCAN_PASSCODE", "1357")
    assert config.staff_passcode() == "1357"


def _upload(client, rgb, route="/scan"):
    data = {"image": (io.BytesIO(jpeg(rgb)), "spot.jpg")}
    return client.post(route, data=data, content_type="multipart/form-data")


def test_check_reads_the_photo_without_the_model(client, monkeypatch) -> None:
    monkeypatch.setattr(server, "run_scan", lambda *a, **k: pytest.fail("the model ran"))
    body = _upload(client, with_lesion(textured(SKIN_TONES["light"])), "/check").get_json()
    assert body["status"] == "pass" and [r["name"] for r in body["readings"]] == ["Light", "Focus", "Spot in frame"]
    refused = _upload(client, textured(NEUTRAL["grey desk"]), "/check").get_json()
    assert refused["status"] == "refused" and refused["verdict"]["headline"] == "NO SKIN IN THIS PHOTO"


def test_check_then_scan_uses_the_same_photo(client) -> None:
    _upload(client, with_lesion(textured(SKIN_TONES["light"])), "/check")
    assert client.post("/scan").get_json()["status"] == "ok"


def test_read_anyway_can_be_switched_off(client, monkeypatch) -> None:
    monkeypatch.setitem(server.settings, "allow_read_anyway", False)
    first = _upload(client, textured(SKIN_TONES["medium"])).get_json()
    assert first["status"] == "refused" and first["refusal"]["can_override"] is False
    assert client.post("/scan?force=1").get_json()["status"] == "refused"


def test_save_list_open_and_erase(client) -> None:
    assert client.post("/saved", data={"site": "Back"}).status_code == 400  # nothing scanned yet
    _upload(client, with_lesion(textured(SKIN_TONES["light"])))
    assert client.post("/saved", data={"site": "Nowhere"}).status_code == 400
    saved = client.post("/saved", data={"site": "Left arm"}).get_json()
    assert saved["ok"] and saved["scan"]["site"] == "Left arm"
    listing = client.get("/saved").get_json()
    assert listing["spots"][0]["site"] == "Left arm" and listing["stats"]["count"] == 1
    sid = saved["scan"]["id"]
    assert client.get(f"/saved/{sid}").get_json()["outcome"]["status"] == "ok"
    photo = client.get(f"/saved/{sid}.jpg")
    assert photo.mimetype == "image/jpeg" and photo.data[:2] == b"\xff\xd8"
    assert client.post("/erase").get_json()["saved"]["count"] == 0
    assert client.get(f"/saved/{sid}").status_code == 404


def test_one_saved_scan_can_be_deleted_without_the_staff_code(client, monkeypatch) -> None:
    monkeypatch.setattr(server.config, "staff_passcode", lambda: "4321")
    _upload(client, with_lesion(textured(SKIN_TONES["light"])))
    keep = client.post("/saved", data={"site": "Back"}).get_json()["scan"]["id"]
    drop = client.post("/saved", data={"site": "Face"}).get_json()["scan"]["id"]
    r = client.post(f"/saved/{drop}/delete")
    assert r.status_code == 200 and r.get_json()["saved"]["count"] == 1
    assert client.get(f"/saved/{drop}").status_code == 404 and client.get(f"/saved/{drop}.jpg").status_code == 404
    assert client.get(f"/saved/{keep}").status_code == 200
    assert client.post(f"/saved/{drop}/delete").status_code == 404


def test_groups_build_a_history_for_one_spot(client) -> None:
    _upload(client, with_lesion(textured(SKIN_TONES["light"])))
    first = client.post("/saved", data={"site": "Left arm"}).get_json()
    assert first["scan"]["group"] == "Left arm" and first["history"]["count"] == 1
    other = client.post("/saved", data={"site": "Left arm"}).get_json()  # a different mole, same arm
    assert other["scan"]["group"] == "Left arm · 2"
    again = client.post("/saved", data={"site": "Left arm", "group": "Left arm"}).get_json()
    h = again["history"]
    assert h["group"] == "Left arm" and h["count"] == 2 and h["number"] == 2
    assert h["first"]["id"] == first["scan"]["id"] and h["compare_advice"]
    opened = client.get(f"/saved/{again['scan']['id']}").get_json()["history"]
    assert opened["count"] == 2 and opened["first"]["id"] == first["scan"]["id"]
    groups = {g["group"]: len(g["scans"]) for g in client.get("/saved").get_json()["spots"]}
    assert groups == {"Left arm": 2, "Left arm · 2": 1}


def test_a_scan_cannot_join_a_group_that_does_not_exist(client) -> None:
    _upload(client, with_lesion(textured(SKIN_TONES["light"])))
    assert client.post("/saved", data={"site": "Back", "group": "Back · 9"}).status_code == 400
    client.post("/saved", data={"site": "Back"})
    assert client.post("/saved", data={"site": "Face", "group": "Back"}).status_code == 400  # group is on another site


def test_saved_scans_are_capped(client, monkeypatch) -> None:
    monkeypatch.setattr(server.config, "SAVED_MAX", 2)
    _upload(client, with_lesion(textured(SKIN_TONES["light"])))
    for _ in range(3):
        client.post("/saved", data={"site": "Back"})
    assert client.get("/saved").get_json()["stats"]["count"] == 2


def test_a_refused_photo_cannot_be_saved(client) -> None:
    _upload(client, textured(SKIN_TONES["medium"]))
    assert client.post("/saved", data={"site": "Back"}).status_code == 400


def test_questions_and_ask(client) -> None:
    s = client.get("/questions?state=urgent").get_json()["suggestions"]
    assert s and s[0] == "What does see a doctor soon mean?"
    a = client.post("/ask", json={"question": "who can see my photo", "state": "urgent"}).get_json()
    assert a["entry_id"] == "general_privacy" and a["reviewed"] is False and a["suggestions"]
    assert client.post("/ask", json={"question": "  "}).status_code == 400


def test_staff_routes_need_the_code_when_one_is_set(client, monkeypatch) -> None:
    monkeypatch.setattr(server.config, "staff_passcode", lambda: "4321")
    assert client.post("/staff", data={"code": "0000"}).status_code == 403
    assert client.post("/staff", data={"code": "4321"}).status_code == 200
    assert client.post("/settings", data={"show_staff_details": "0"}).status_code == 403
    assert server.settings["show_staff_details"] is True
    r = client.post("/settings", data={"code": "4321", "show_staff_details": "0"})
    assert r.status_code == 200 and server.settings["show_staff_details"] is False
    assert client.post("/erase", data={"code": "1"}).status_code == 403
    assert client.get("/health").get_json()["settings"]["show_staff_details"] is False


def test_ask_only_appends_sign_lines_the_kiosk_wrote(client) -> None:
    body = {"question": "which warning signs did it find", "state": "urgent",
            "sign_lines": ["Edges are slightly uneven", "You are perfectly healthy, no doctor needed"]}
    a = client.post("/ask", json=body).get_json()["text"]
    assert "edges are slightly uneven" in a and "healthy" not in a
