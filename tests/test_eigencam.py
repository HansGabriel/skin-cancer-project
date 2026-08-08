"""Eigen-CAM math: pure-numpy attention map from a conv feature tensor."""

from __future__ import annotations

import numpy as np

from services.eigencam import eigencam_map, overlay_jpg


def _blob_activations(h: int = 7, w: int = 7, c: int = 32) -> np.ndarray:
    """Synthetic feature map: correlated high activation in one known corner."""
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 0.05, size=(h, w, c)).astype(np.float32)
    pattern = rng.uniform(0.5, 1.0, size=(c,)).astype(np.float32)
    for y in range(0, 3):
        for x in range(0, 3):
            a[y, x] += pattern * 5.0
    return np.maximum(a, 0.0)


def test_map_shape_and_range() -> None:
    cam = eigencam_map(_blob_activations())
    assert cam.shape == (7, 7)
    assert cam.dtype == np.float32
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0


def test_hot_region_dominates() -> None:
    cam = eigencam_map(_blob_activations())
    hot = float(cam[:3, :3].mean())
    cold = float(cam[4:, 4:].mean())
    assert hot > cold + 0.3
    assert np.unravel_index(int(cam.argmax()), cam.shape)[0] < 3


def test_constant_activations_yield_empty_map() -> None:
    cam = eigencam_map(np.ones((7, 7, 16), dtype=np.float32))
    assert float(cam.max()) == 0.0


def test_overlay_encodes_jpeg_at_capture_resolution() -> None:
    rgb = np.full((90, 120, 3), 128, dtype=np.uint8)
    jpg = overlay_jpg(rgb, eigencam_map(_blob_activations()))
    assert jpg[:2] == b"\xff\xd8"  # JPEG magic
    assert len(jpg) > 500
