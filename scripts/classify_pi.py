"""Standalone Raspberry Pi capture + inference script."""

from __future__ import annotations

import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2


MODEL = "skin_classifier.tflite"
LABELS_FILE = "labels.txt"
IMAGE_SIZE = 224

RECOMMENDATIONS = {
    "benign": {
        "icon": "🟢",
        "urgency": "LOW CONCERN",
        "action": "No immediate action needed. Continue regular self-checks.",
    },
    "pre_cancerous": {
        "icon": "🟠",
        "urgency": "IMPORTANT",
        "action": "Schedule a dermatology consultation within 1 month.",
    },
    "malignant": {
        "icon": "🔴",
        "urgency": "URGENT",
        "action": "See a dermatologist within 1-2 weeks.",
    },
}


def capture_rgb() -> np.ndarray:
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration(main={"size": (1024, 1024)}))
    picam2.start()
    time.sleep(1.0)
    image = picam2.capture_array()
    picam2.stop()
    return image


def preprocess(image_rgb: np.ndarray, input_details: dict) -> np.ndarray:
    image = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
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
    if output_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_details["quantization"]
        if scale == 0:
            raise ValueError("Invalid output quantization scale=0")
        return (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)


def main() -> None:
    labels = [line.strip() for line in open(LABELS_FILE, "r", encoding="utf-8") if line.strip()]

    interpreter = tflite.Interpreter(model_path=MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("Input details:", input_details)
    print("Output details:", output_details)
    print("Capturing image...")
    image_rgb = capture_rgb()
    cv2.imwrite("last_capture.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    input_tensor = preprocess(image_rgb, input_details)
    start = time.time()
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    probs = dequantize_output(output, output_details)
    elapsed = time.time() - start

    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    recommendation = RECOMMENDATIONS[pred_label]

    print("\n" + "=" * 60)
    print(f"{recommendation['icon']} {recommendation['urgency']}")
    print(f"Detected: {pred_label.upper()} ({confidence:.1f}% confidence)")
    print(f"Action: {recommendation['action']}")
    print("\nAll class probabilities:")
    for i, label in enumerate(labels):
        print(f"  {label:>15}: {float(probs[i]) * 100.0:5.1f}%")
    print(f"\nInference time: {elapsed:.2f}s")
    print("=" * 60)
    print("For screening only - not a medical diagnosis.")


if __name__ == "__main__":
    main()
