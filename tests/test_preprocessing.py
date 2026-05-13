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

    def test_int8_input_quantize_dequantize_roundtrip(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
        # Scale / zero_point chosen so [0, 255] maps into int8 without saturation.
        scale, zero_point = 1.0, -128
        in_details = {"dtype": np.int8, "quantization": (scale, zero_point)}
        out_details = {"dtype": np.int8, "quantization": (scale, zero_point)}
        q = preprocessing.to_input_tensor(rgb, in_details)
        self.assertEqual(q.dtype, np.int8)
        expected_float = preprocessing.resize_to_rgb_224(rgb).astype(np.float32)
        back = preprocessing.dequantize_output(q, out_details)
        np.testing.assert_allclose(back[0], expected_float, atol=1e-5)

    def test_uint8_input_quantize_dequantize_roundtrip(self) -> None:
        rgb = np.random.default_rng(1).integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
        scale, zero_point = 1.0, 0.0
        in_details = {"dtype": np.uint8, "quantization": (scale, zero_point)}
        out_details = {"dtype": np.uint8, "quantization": (scale, zero_point)}
        q = preprocessing.to_input_tensor(rgb, in_details)
        self.assertEqual(q.dtype, np.uint8)
        expected_float = preprocessing.resize_to_rgb_224(rgb).astype(np.float32)
        back = preprocessing.dequantize_output(q, out_details)
        np.testing.assert_allclose(back[0], expected_float, atol=1.0)


if __name__ == "__main__":
    unittest.main()
