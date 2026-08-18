"""The app must import cleanly on any interpreter the deployment might pick.

Two Streamlit Cloud outages came from module-level imports that were fine
locally and fatal there, each reported as a redacted ImportError pointing at a
module that was not the cause (Streamlit shows the last frame it cleared, not
the one that raised). Both were only visible by importing the whole graph.

This walks every module under frontend/ and backend/ and imports it. It runs on
the local interpreter; `tests/test_python_compat.py` covers the version-specific
names that made the older interpreter fail.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

for path in (ROOT, ROOT / "frontend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _module_names() -> list[str]:
    names: list[str] = []
    for pkg, base in (("frontend", ROOT / "frontend"), ("backend", ROOT / "backend")):
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(base if pkg == "frontend" else ROOT)
            name = str(rel.with_suffix("")).replace("/", ".")
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            # app.py calls st.set_page_config at import; covered by the AppTest suite.
            if name == "app":
                continue
            names.append(name)
    return names


@pytest.mark.parametrize("module", _module_names())
def test_module_imports_cleanly(module: str) -> None:
    """A module-level import error here is a Cloud outage there."""
    importlib.import_module(module)


def test_every_name_app_imports_actually_exists() -> None:
    """Catches a `from x import y` where y was renamed or never landed.

    The second outage looked exactly like this — `from backend.assistant import
    (..., is_lesion_specific, ...)` — so the check is cheap insurance even
    though that instance turned out to be a stale deploy.
    """
    import ast

    missing: list[str] = []
    for path in sorted((ROOT / "frontend").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if node.module.split(".")[0] not in ("backend", "services", "components", "views", "theme"):
                continue
            try:
                mod = importlib.import_module(node.module)
            except ImportError:  # pragma: no cover - the parametrised test reports it
                continue
            for alias in node.names:
                if alias.name != "*" and not hasattr(mod, alias.name):
                    missing.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} imports "
                        f"`{alias.name}` from {node.module}, which does not define it"
                    )
    assert not missing, "\n".join(missing)
