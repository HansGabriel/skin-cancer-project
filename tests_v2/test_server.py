"""The Flask routes, with the camera absent (as on a laptop) and faked (as on the Pi)."""

from __future__ import annotations

import io

import pytest

from conftest import NEUTRAL, SKIN_TONES, jpeg, textured, with_lesion
from kiosk import camera, server


@pytest.fixture()
def client():
    server._last_capture = None
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
    assert client.post("/scan").status_code == 400


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
    assert client.post("/capture").status_code == 404
    assert client.get("/preview.mjpg").status_code == 404


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
    monkeypatch.setattr(server, "QUIT_FLAG", flag)
    assert client.post("/quit").get_json()["ok"]
    assert flag.read_text() == "quit"
