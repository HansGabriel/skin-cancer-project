"""The Flask routes, with the camera absent (as on a laptop) and faked (as on the Pi)."""

from __future__ import annotations

import io

import pytest

from conftest import NEUTRAL, SKIN_TONES, jpeg, textured, with_lesion
from kiosk import camera, server


@pytest.fixture()
def client(monkeypatch):
    server.last_photo.forget()
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
