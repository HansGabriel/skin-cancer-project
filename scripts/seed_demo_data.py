"""Seed a freshly deployed DermaScan DB with one demo folder + case + two scans.

Idempotent: skips if a folder named ``Demo cases`` already exists. Designed to
be run once after a Streamlit Cloud deploy so the History screen isn't empty.

Usage:
    python scripts/seed_demo_data.py [--data-dir PATH] [--model PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRONTEND))

DEMO_FOLDER_NAME = "Demo cases"
DEMO_CASE_NAME = "Forearm mole #1"
DEMO_IMAGES = ["benign_demo.jpg", "pre_cancerous_demo.jpg"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Override DERMASCAN_DATA_DIR for this run.")
    parser.add_argument(
        "--model",
        default=os.environ.get("SKIN_MODEL_PATH", str(ROOT / "models" / "skin_classifier.tflite")),
    )
    parser.add_argument(
        "--labels",
        default=os.environ.get("SKIN_LABELS_PATH", str(ROOT / "models" / "labels.txt")),
    )
    args = parser.parse_args()

    if args.data_dir:
        os.environ["DERMASCAN_DATA_DIR"] = args.data_dir

    from backend import get_backend
    from services.scan_flow import run_scan_and_store
    from services.storage import get_storage

    store = get_storage()
    if any(f.name == DEMO_FOLDER_NAME for f in store.list_folders()):
        print(f"Demo folder '{DEMO_FOLDER_NAME}' already exists — skipping.")
        return 0

    folder = store.create_folder(DEMO_FOLDER_NAME)
    case = store.create_case(folder.id, DEMO_CASE_NAME, body_site="arm", notes="Seeded demo case.")
    print(f"Created folder={folder.id} case={case.id}")

    backend = get_backend(kind="mock", model_path=args.model, labels_path=args.labels)
    samples_dir = ROOT / "samples"
    for fname in DEMO_IMAGES:
        path = samples_dir / fname
        if not path.exists():
            print(f"WARNING: missing sample {path} — skipped.")
            continue
        os.environ["SKIN_TTA"] = "0"
        run_scan_and_store(
            backend,
            path.read_bytes(),
            pixels_per_mm=10.0,
            strict_quality=False,
            keras_path="",
            case_id=case.id,
        )
        print(f"  + scan from {fname}")

    print("Seed complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
