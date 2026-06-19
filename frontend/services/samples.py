"""Locate bundled demo/sample images for the mock backend picker."""

from __future__ import annotations

from pathlib import Path

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_sample_paths(root: Path) -> list[tuple[str, Path]]:
    """Return (label, path) pairs for sample images under the project root."""
    items: list[tuple[str, Path]] = []
    samples_dir = root / "samples"
    if samples_dir.is_dir():
        for p in sorted(samples_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                items.append((p.name, p))
    test_samples = root / "datasets" / "ham10000" / "test_samples"
    if test_samples.is_dir():
        for p in sorted(test_samples.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                items.append((f"ham10000/{p.name}", p))
    return items
