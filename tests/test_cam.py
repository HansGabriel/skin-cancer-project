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


def _fresh(monkeypatch) -> None:
    monkeypatch.setattr(cam, "_cam", None)
    monkeypatch.setattr(cam, "_cam_tried", False)


def test_the_deployed_classifier_gives_a_heatmap_without_a_cam_file(monkeypatch, tmp_path) -> None:
    """No fifth file needed: the 7x7x1280 map is read out of skin_classifier.tflite itself."""
    monkeypatch.setattr(config, "CAM_PATH", tmp_path / "missing.tflite")
    _fresh(monkeypatch)
    c = cam.get_cam()
    assert c is not None and c.source == "classifier"
    rgb = with_lesion(textured(SKIN_TONES["light"], size=320))
    m = c.attention(rgb)
    assert m.shape == (7, 7) and 0.0 <= m.min() and m.max() == 1.0
    assert m[2:5, 2:5].max() == 1.0  # a centred mole: the hottest cell is in the middle
    data = c.heatmap(rgb)
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == rgb.shape[:2]


def test_a_cam_file_from_other_weights_falls_back_to_the_classifier(monkeypatch, tmp_path) -> None:
    """The sha in skin_classifier_cam.json must be the deployed classifier's; a stranger's file is skipped."""
    fake = tmp_path / "skin_classifier_cam.tflite"
    fake.write_bytes(b"not a model")
    meta = tmp_path / "skin_classifier_cam.json"
    meta.write_text(json.dumps({"classifier_sha256": "0" * 64}))
    monkeypatch.setattr(config, "CAM_PATH", fake)
    monkeypatch.setattr(config, "CAM_META_PATH", meta)
    _fresh(monkeypatch)
    assert not cam.cam_matches_classifier(meta, config.MODEL_PATH)
    c = cam.get_cam()
    assert c is not None and c.source == "classifier"
    meta.write_text(json.dumps({"classifier_sha256": cam.sha256_of(config.MODEL_PATH)}))
    assert cam.cam_matches_classifier(meta, config.MODEL_PATH)


def test_no_model_at_all_means_no_cam(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "CAM_PATH", tmp_path / "missing.tflite")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing_classifier.tflite")
    _fresh(monkeypatch)
    assert cam.get_cam() is None


def test_feature_tensor_prefers_the_named_layer_then_the_deepest_map() -> None:
    details = [
        {"index": 3, "name": "block7/x", "shape": np.array([1, 7, 7, 320])},
        {"index": 9, "name": "functional_1/top_activation_1/mul", "shape": np.array([1, 7, 7, 1280])},
        {"index": 12, "name": "pool", "shape": np.array([1, 1280])},
    ]
    assert cam.feature_tensor(details)["index"] == 9
    assert cam.feature_tensor(details[:1])["index"] == 3
    assert cam.feature_tensor(details[2:]) is None
    q = {"index": 0, "quantization": (0.5, -128)}
    assert cam.dequantise(np.array([-128, -126], np.int8), q).tolist() == [0.0, 1.0]
