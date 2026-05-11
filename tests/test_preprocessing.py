"""Unit tests for pure preprocessing (no TFLite required)."""

from __future__ import annotations

import unittest

import numpy as np

from backend import preprocessing


class TestPreprocessing(unittest.TestCase):
    def test_resize_to_rgb_224_shape(self) -> None:
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        out = preprocessing.resize_to_rgb_224(img)
        self.assertEqual(out.shape, (224, 224, 3))

    def test_float32_input_tensor_roundtrip(self) -> None:
        rgb = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        details = {"dtype": np.float32, "quantization": (0.0, 0)}
        tensor = preprocessing.to_input_tensor(rgb, details)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        raw = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        out = preprocessing.dequantize_output(raw, {"dtype": np.float32, "quantization": (0.0, 0)})
        np.testing.assert_allclose(out, raw, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
