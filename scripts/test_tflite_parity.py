"""Compare Keras and TFLite predictions on identical images.

Usage:
  python scripts/test_tflite_parity.py --images_dir datasets/ham10000/test_samples
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras_model", default="models/skin_classifier_full.keras")
    parser.add_argument("--tflite_model", default="models/skin_classifier.tflite")
    parser.add_argument("--labels", default="models/labels.txt")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def preprocess(image_bgr: np.ndarray, img_size: int) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (img_size, img_size))
    image_float = image_resized.astype(np.float32)
    return np.expand_dims(image_float, axis=0)


def tflite_predict(interpreter: tf.lite.Interpreter, x: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    if input_details["dtype"] == np.float32:
        x_input = x.astype(np.float32)
    elif input_details["dtype"] == np.uint8:
        if in_scale == 0:
            raise ValueError("TFLite input quantization scale is 0 for uint8 model.")
        x_input = np.clip(np.round(x / in_scale + in_zero), 0, 255).astype(np.uint8)
    elif input_details["dtype"] == np.int8:
        if in_scale == 0:
            raise ValueError("TFLite input quantization scale is 0 for int8 model.")
        x_input = np.clip(np.round(x / in_scale + in_zero), -128, 127).astype(np.int8)
    else:
        raise TypeError(f"Unsupported TFLite input dtype: {input_details['dtype']}")

    interpreter.set_tensor(input_details["index"], x_input)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details["index"])[0]

    if output_details["dtype"] in (np.uint8, np.int8):
        if out_scale == 0:
            raise ValueError("TFLite output quantization scale is 0.")
        y = (y.astype(np.float32) - out_zero) * out_scale
    else:
        y = y.astype(np.float32)
    return y


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels))
    keras_model = tf.keras.models.load_model(args.keras_model)
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()

    image_paths = sorted(
        p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    print(f"TFLite input details: {interpreter.get_input_details()[0]}")
    print(f"TFLite output details: {interpreter.get_output_details()[0]}\n")

    same_top1 = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[skip] {path.name}: failed to read")
            continue

        x = preprocess(image, args.img_size)
        keras_probs = keras_model.predict(x, verbose=0)[0].astype(np.float32)
        tflite_probs = tflite_predict(interpreter, x)

        k_idx = int(np.argmax(keras_probs))
        t_idx = int(np.argmax(tflite_probs))
        if k_idx == t_idx:
            same_top1 += 1

        print(
            f"{path.name:35s} | "
            f"Keras: {labels[k_idx]:15s} ({keras_probs[k_idx]*100:5.1f}%) | "
            f"TFLite: {labels[t_idx]:15s} ({tflite_probs[t_idx]*100:5.1f}%)"
        )

    total = len(image_paths)
    print(f"\nTop-1 agreement: {same_top1}/{total} = {same_top1/total:.1%}")


if __name__ == "__main__":
    main()
