"""Quick Keras inference sanity check on local images.

Usage:
  python scripts/test_inference_pc.py --images_dir datasets/ham10000/test_samples
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/skin_classifier_full.keras",
        help="Path to .keras model",
    )
    parser.add_argument(
        "--labels",
        default="models/labels.txt",
        help="Path to labels file (one label per line)",
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory containing .jpg/.jpeg/.png files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of images to evaluate",
    )
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def preprocess_for_efficientnet(image_bgr: np.ndarray, img_size: int) -> np.ndarray:
    # For TF/Keras EfficientNet in this notebook setup, keep float pixels in [0, 255].
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (img_size, img_size))
    image_float = image_resized.astype(np.float32)
    return np.expand_dims(image_float, axis=0)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    labels_path = Path(args.labels)
    images_dir = Path(args.images_dir)

    model = tf.keras.models.load_model(model_path)
    labels = load_labels(labels_path)

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.limit]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"Model: {model_path}")
    print(f"Labels: {labels}")
    print(f"Images: {len(image_paths)} from {images_dir}\n")

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[skip] {path.name}: failed to read")
            continue

        x = preprocess_for_efficientnet(image, args.img_size)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0
        print(f"{path.name:35s} -> {pred_label:15s} ({confidence:5.1f}%)")


if __name__ == "__main__":
    main()
