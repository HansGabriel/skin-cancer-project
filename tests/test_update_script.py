"""scripts/update.sh against a throwaway git repo with a bare "origin".

The clone has no venv and no tests folder, so the pip and pytest steps are skipped
and only the git logic is exercised: refuse off main, refuse local edits, say
"already up to date", fast-forward, and record one line in last_update.txt.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update.sh"
pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash")


def _git(cwd: Path, *args: str) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def repos(tmp_path: Path):
    origin, work, clone = tmp_path / "origin.git", tmp_path / "work", tmp_path / "clone"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    (work / "README.md").write_text("one\n")
    _git(work, "add", "."), _git(work, "commit", "-qm", "one"), _git(work, "push", "-q", "origin", "main")
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    return origin, work, clone


def _update(clone: Path, state: Path, repo_url: str | None = None) -> str:
    """Run the script against `repo_url` (default: the clone's own bare origin) and return its message."""
    url = repo_url or _git(clone, "remote", "get-url", "origin")
    out = subprocess.run(["bash", str(SCRIPT), str(clone), str(state), url], capture_output=True, text=True, check=True)
    line = (state / "last_update.txt").read_text().strip()
    assert out.stdout.strip() == line
    return line.split(" · ", 1)[1]


def test_already_up_to_date_then_fast_forward(repos, tmp_path) -> None:
    _, work, clone = repos
    state = tmp_path / "state"
    assert _update(clone, state).startswith("already up to date (")

    (work / "README.md").write_text("two\n")
    _git(work, "commit", "-qam", "two"), _git(work, "push", "-q", "origin", "main")
    msg = _update(clone, state)
    assert msg.startswith("updated ") and _git(clone, "rev-parse", "HEAD") == _git(work, "rev-parse", "HEAD")


def test_fetch_goes_to_the_named_url_not_to_origin(repos, tmp_path) -> None:
    """The Pi's origin is SSH and cannot be used from the launcher; the script fetches the URL it is given."""
    origin, work, clone = repos
    state = tmp_path / "state"
    _git(clone, "remote", "set-url", "origin", "git@github.com:nobody/nowhere.git")
    (work / "README.md").write_text("two\n")
    _git(work, "commit", "-qam", "two"), _git(work, "push", "-q", "origin", "main")
    assert _update(clone, state, str(origin)).startswith("updated ")


def test_refuses_local_edits_and_other_branches(repos, tmp_path) -> None:
    _, _, clone = repos
    state = tmp_path / "state"
    (clone / "README.md").write_text("edited on the Pi\n")
    assert _update(clone, state).startswith("refused: local edits in README.md")
    _git(clone, "checkout", "-q", "--", "README.md")

    _git(clone, "checkout", "-qb", "feat/x")
    assert _update(clone, state).startswith("refused: the Pi is on branch 'feat/x'")


def test_no_remote_is_a_fetch_failure_not_a_wifi_one(repos, tmp_path) -> None:
    origin, _, clone = repos
    url = str(origin)
    shutil.rmtree(origin)
    msg = _update(clone, tmp_path / "state", url)
    assert msg.startswith("failed: could not fetch from GitHub") and "wifi" not in msg


def test_no_internet_is_named_as_such(repos, tmp_path) -> None:
    """An HTTPS address that does not answer means no network, and says so before any git runs."""
    _, _, clone = repos
    state = tmp_path / "state"
    msg = _update(clone, state, "http://127.0.0.1:9/skin-cancer-project.git")
    assert msg.startswith("failed: no internet")
    assert "fetch" not in (state / "update.log").read_text()
