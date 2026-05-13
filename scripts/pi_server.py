"""Flask API on Raspberry Pi: capture image + run TFLite + return JSON."""

from __future__ import annotations

import base64
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, jsonify, request
from picamera2 import Picamera2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "skin_classifier.tflite"
LABELS_PATH = BASE_DIR / "labels.txt"
DB_PATH = BASE_DIR / "pi_scans.sqlite"
IMAGE_SIZE = 224

# Main stream must be explicit RGB so preprocessing and JPEG encoding match training.
PICAM_STILL_MAIN = {"size": (1024, 1024), "format": "RGB888"}

# Keys must match exactly one line per row in labels.txt (3-class deployment contract).
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


def load_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
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
    cam.configure(cam.create_still_configuration(main=PICAM_STILL_MAIN))
    try:
        cam.start()
        time.sleep(1.0)
        return cam.capture_array()
    finally:
        try:
            cam.stop()
        except Exception:
            pass


def decode_upload_to_rgb(file_storage) -> np.ndarray:
    """Decode multipart upload (JPEG/PNG bytes) to RGB uint8 HWC."""
    raw = file_storage.read()
    if not raw:
        raise ValueError("Empty upload body")
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image (use JPEG or PNG)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_scan_from_rgb(
    image_rgb: np.ndarray,
    *,
    ts_iso: str,
) -> dict:
    """RGB uint8 HWC -> infer -> success dict for jsonify (no DB insert)."""
    input_tensor = preprocess(image_rgb, input_details)

    t0 = time.perf_counter()
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    inference_ms = int((time.perf_counter() - t0) * 1000)

    probs = dequantize_output(output, output_details)
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    recommendation = RECOMMENDATIONS[pred_label]
    probs_pct = {labels[i]: float(probs[i]) * 100.0 for i in range(len(labels))}

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO scans (ts_iso, label, confidence, inference_ms, probs_json) VALUES (?,?,?,?,?)",
            (ts_iso, pred_label, confidence, inference_ms, json.dumps(probs_pct)),
        )
        conn.commit()

    return {
        "label": pred_label,
        "confidence": confidence,
        "icon": recommendation["icon"],
        "urgency": recommendation["urgency"],
        "action": recommendation["action"],
        "probs": probs_pct,
        "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "inference_ms": inference_ms,
        "timestamp": ts_iso,
    }


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_iso TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                inference_ms INTEGER NOT NULL,
                probs_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


init_db()

labels = load_labels(LABELS_PATH)
if set(labels) != set(RECOMMENDATIONS):
    raise RuntimeError(
        f"labels.txt classes {set(labels)} must match RECOMMENDATIONS keys {set(RECOMMENDATIONS)}"
    )

interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH.name, "labels": LABELS_PATH.name})


@app.get("/log")
def get_log():
    if not DB_PATH.exists():
        return jsonify([])
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT ts_iso, label, confidence, inference_ms, probs_json "
            "FROM scans ORDER BY id DESC LIMIT 500"
        )
        rows = []
        for row in cur.fetchall():
            rows.append(
                {
                    "ts_iso": row["ts_iso"],
                    "backend_id": "pi",
                    "label": row["label"],
                    "confidence": row["confidence"],
                    "probs": json.loads(row["probs_json"]),
                    "inference_ms": row["inference_ms"],
                }
            )
        return jsonify(rows)
    finally:
        conn.close()


@app.post("/scan")
def scan():
    ts_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    try:
        upload = request.files.get("image")
        if upload is not None and upload.filename:
            try:
                image_rgb = decode_upload_to_rgb(upload)
            except ValueError as exc:
                return jsonify({"status": "error", "reason": str(exc)}), 400
        else:
            try:
                image_rgb = capture_rgb()
            except Exception as exc:  # noqa: BLE001
                return jsonify(
                    {
                        "status": "error",
                        "reason": f"Camera capture failed: {exc!s}",
                    }
                ), 503

        payload = run_scan_from_rgb(image_rgb, ts_iso=ts_iso)
        return jsonify(payload)
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        return jsonify({"status": "error", "reason": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "reason": f"Inference failed: {exc!s}"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
