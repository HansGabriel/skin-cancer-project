"""Pure NumPy / OpenCV transforms for TFLite I/O (no TFLite import).

Training (see ``notebooks/train_skin_classifier.ipynb``) feeds EfficientNetB0 with
linear RGB values in ``[0, 255]`` (``uint8`` in ``tf.data``, then
``keras.applications.efficientnet.preprocess_input``; on TF 2.21 this is effectively
identity while the base model applies internal rescaling). TFLite inference here must
match the **exported graph input** (float ``[0, 255]`` before per-tensor quantization).
Do not add an extra ``(x / 127.5) - 1.0`` unless the trained/exported model expects it.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = 224


def resize_to_rgb_224(image_rgb: np.ndarray) -> np.ndarray:
    """Resize RGB uint8 image to ``IMAGE_SIZE`` square."""
    return cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def to_input_tensor(image_rgb: np.ndarray, input_details: dict) -> np.ndarray:
    """Match ``scripts/pi_server.py`` / ``classify_pi.py`` preprocessing."""
    image = resize_to_rgb_224(image_rgb)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    dtype = input_details["dtype"]
    scale, zero_point = input_details["quantization"]

    if dtype == np.float32:
        return image
    if dtype == np.uint8:
        if scale == 0:
            raise ValueError("Invalid uint8 quantization scale=0")
        return np.clip(np.round(image / scale + zero_point), 0, 255).astype(np.uint8)
    if dtype == np.int8:
        if scale == 0:
            raise ValueError("Invalid int8 quantization scale=0")
        return np.clip(np.round(image / scale + zero_point), -128, 127).astype(np.int8)
    raise TypeError(f"Unsupported input dtype: {dtype}")


def dequantize_output(output: np.ndarray, output_details: dict) -> np.ndarray:
    """Dequantize model output to float32 probabilities / logits."""
    if output_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_details["quantization"]
        if scale == 0:
            raise ValueError("Invalid output quantization scale=0")
        return (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)
