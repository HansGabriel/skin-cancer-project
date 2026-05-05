"""Raspberry Pi file-based TFLite inference before camera debugging."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="skin_classifier.tflite")
    parser.add_argument("--labels", default="labels.txt")
    parser.add_argument("--samples_dir", default="samples")
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def preprocess(image_bgr: np.ndarray, img_size: int, input_details: dict) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image_rgb, (img_size, img_size)).astype(np.float32)
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
    args = parse_args()
    labels = [line.strip() for line in Path(args.labels).read_text().splitlines() if line.strip()]

    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("Input details:", input_details)
    print("Output details:", output_details)

    samples_dir = Path(args.samples_dir)
    files = sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise FileNotFoundError(f"No sample images found in {samples_dir}")

    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[skip] {path.name}: failed to read")
            continue

        x = preprocess(image, args.img_size, input_details)
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details["index"])[0]
        probs = dequantize_output(y, output_details)

        idx = int(np.argmax(probs))
        print(f"{path.name:30s} -> {labels[idx]:15s} ({float(probs[idx]) * 100.0:5.1f}%)")


if __name__ == "__main__":
    main()
