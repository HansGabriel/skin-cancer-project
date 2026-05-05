"""Flask API on Raspberry Pi: capture image + run TFLite + return JSON."""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, jsonify
from picamera2 import Picamera2


MODEL_PATH = "skin_classifier.tflite"
LABELS_PATH = "labels.txt"
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


def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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


def capture_rgb() -> np.ndarray:
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (1024, 1024)}))
    cam.start()
    time.sleep(1.0)
    image_rgb = cam.capture_array()
    cam.stop()
    return image_rgb


app = Flask(__name__)
labels = load_labels(LABELS_PATH)
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


@app.post("/scan")
def scan():
    image_rgb = capture_rgb()
    input_tensor = preprocess(image_rgb, input_details)

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    probs = dequantize_output(output, output_details)

    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    recommendation = RECOMMENDATIONS[pred_label]

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")

    return jsonify(
        {
            "label": pred_label,
            "confidence": confidence,
            "icon": recommendation["icon"],
            "urgency": recommendation["urgency"],
            "action": recommendation["action"],
            "probs": {labels[i]: float(probs[i]) * 100.0 for i in range(len(labels))},
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
