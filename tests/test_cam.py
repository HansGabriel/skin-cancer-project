"""Eigen-CAM pieces without a CAM model file: the map, the overlay, and the guard."""

from __future__ import annotations

import json

import cv2
import numpy as np

from conftest import SKIN_TONES, textured, with_lesion
from dermascan import cam, config


def test_eigencam_map_points_at_the_hot_cell() -> None:
    rng = np.random.default_rng(0)
    act = rng.normal(0, 0.05, size=(7, 7, 16)).astype(np.float32)
    act[2, 5, :] += 3.0  # one strongly active position
    m = cam.eigencam_map(act)
    assert m.shape == (7, 7) and m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() == 1.0
    assert np.unravel_index(int(m.argmax()), m.shape) == (2, 5)


def test_flat_activations_give_an_empty_map() -> None:
    assert not cam.eigencam_map(np.ones((7, 7, 8), np.float32)).any()


def test_overlay_is_a_jpeg_the_size_of_the_photo() -> None:
    rgb = with_lesion(textured(SKIN_TONES["light"], size=320))
    m = np.zeros((7, 7), np.float32)
    m[3, 3] = 1.0
    data = cam.overlay_jpg(rgb, m)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[:2] == rgb.shape[:2]


def test_no_cam_file_means_no_cam(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "CAM_PATH", tmp_path / "missing.tflite")
    monkeypatch.setattr(cam, "_cam", None)
    monkeypatch.setattr(cam, "_cam_tried", False)
    assert cam.get_cam() is None


def test_a_cam_file_from_other_weights_is_refused(monkeypatch, tmp_path) -> None:
    """The sha in skin_classifier_cam.json must be the deployed classifier's."""
    fake = tmp_path / "skin_classifier_cam.tflite"
    fake.write_bytes(b"not a model")
    meta = tmp_path / "skin_classifier_cam.json"
    meta.write_text(json.dumps({"classifier_sha256": "0" * 64}))
    monkeypatch.setattr(config, "CAM_PATH", fake)
    monkeypatch.setattr(config, "CAM_META_PATH", meta)
    monkeypatch.setattr(cam, "_cam", None)
    monkeypatch.setattr(cam, "_cam_tried", False)
    assert not cam.cam_matches_classifier(meta, config.MODEL_PATH)
    assert cam.get_cam() is None
    meta.write_text(json.dumps({"classifier_sha256": cam.sha256_of(config.MODEL_PATH)}))
    assert cam.cam_matches_classifier(meta, config.MODEL_PATH)
