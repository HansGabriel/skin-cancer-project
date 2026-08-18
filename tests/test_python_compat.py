"""The deployed interpreter is not guaranteed to be the local one.

Streamlit Community Cloud picks its own Python unless ``runtime.txt`` pins one,
and a 3.11-only import there fails the whole app at start-up with a redacted
ImportError that points at an unrelated module — which is exactly how this cost
an evening. ``runtime.txt`` is the fix; these tests are the belt to its braces.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Names that only exist in `typing` from the given version onward.
_TYPING_MIN_VERSION = {
    "NotRequired": (3, 11),
    "Required": (3, 11),
    "Self": (3, 11),
    "LiteralString": (3, 11),
    "TypeVarTuple": (3, 11),
    "Unpack": (3, 11),
    "assert_never": (3, 11),
    "assert_type": (3, 11),
    "dataclass_transform": (3, 11),
    "override": (3, 12),
    "TypeAliasType": (3, 12),
}


def _python_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for pkg in ("frontend", "backend"):
        out += [p for p in (ROOT / pkg).rglob("*.py") if "__pycache__" not in p.parts]
    return out


def _guarded(node: ast.ImportFrom, tree: ast.Module) -> bool:
    """True when the import sits inside a try/except ImportError shim."""
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.Try):
            continue
        if node not in ast.walk(parent):
            continue
        for handler in parent.handlers:
            names = []
            if isinstance(handler.type, ast.Name):
                names = [handler.type.id]
            elif isinstance(handler.type, ast.Tuple):
                names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
            if "ImportError" in names or handler.type is None:
                return True
    return False


def test_no_unguarded_version_specific_typing_imports() -> None:
    """A 3.11-only name imported bare from `typing` breaks older deployments."""
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "typing":
                continue
            for alias in node.names:
                if alias.name in _TYPING_MIN_VERSION and not _guarded(node, tree):
                    ver = ".".join(map(str, _TYPING_MIN_VERSION[alias.name]))
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} imports "
                        f"`{alias.name}` from typing ({ver}+) without a fallback"
                    )
    assert not offenders, "\n".join(offenders)


def test_runtime_txt_pins_a_python_streamlit_cloud_supports() -> None:
    """Without this file Cloud chooses, and it has chosen wrong before."""
    runtime = ROOT / "runtime.txt"
    assert runtime.is_file(), "runtime.txt is missing — Cloud will pick its own Python"
    pin = runtime.read_text().strip()
    assert pin.startswith("python-"), f"unexpected runtime.txt contents: {pin!r}"
    major, minor = (int(x) for x in pin.removeprefix("python-").split(".")[:2])
    # 3.11+ so the typing names above resolve natively; <=3.13 is what Cloud offers.
    assert (3, 11) <= (major, minor) <= (3, 13), f"{pin} is outside Cloud's supported range"
