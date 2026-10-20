"""Real photos through the real path: every file in tests/fixtures/gate/{refuse,pass}.

The synthetic frames in test_gate.py check the maths; these check the camera. The
folders may be empty in a fresh checkout (no faces of real people in the repo, see
tests/fixtures/gate/README.md): pytest then skips the empty half on its own. The
gitignored captures/{refuse,pass} folders are scanned too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dermascan import scan

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = {
    "refuse": [ROOT / "tests/fixtures/gate/refuse", ROOT / "captures/refuse"],
    "pass": [ROOT / "tests/fixtures/gate/pass", ROOT / "captures/pass"],
}
EXTENSIONS = (".jpg", ".jpeg", ".png")


def _photos(kind: str) -> list[Path]:
    return sorted(p for folder in FOLDERS[kind] if folder.is_dir() for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


@pytest.mark.parametrize("path", _photos("refuse"), ids=lambda p: p.name)
def test_a_photo_that_is_not_one_spot_on_skin_is_refused(path: Path) -> None:
    checked = scan.check_photo(path.read_bytes())
    assert checked.report is not None, f"{path.name}: not a picture"
    assert checked.report.refusal is not None, f"{path.name} passed: {checked.report.measured}"


@pytest.mark.parametrize("path", _photos("pass"), ids=lambda p: p.name)
def test_a_real_lesion_close_up_passes(path: Path) -> None:
    checked = scan.check_photo(path.read_bytes())
    assert checked.report is not None, f"{path.name}: not a picture"
    r = checked.report.refusal
    assert r is None, f"{path.name} refused as {r.code}: {checked.report.measured}"
