"""`scripts/check-knowledge-graph.sh` in each mode, in a scratch git repo, with stdin held open."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-knowledge-graph.sh"
CODE = "src/aigent/a.py"
GRAPH = "docs/knowledge-graph.md"
NOTES = "README.md"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    for path in (CODE, GRAPH, NOTES):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("first\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


@pytest.mark.parametrize(
    ("mode", "staged", "unstaged", "code"),
    [
        pytest.param("local", [], [CODE], 1, id="local: code changed alone"),
        pytest.param("local", [CODE], [GRAPH], 0, id="local: code and graph, staged or not"),
        pytest.param("local", [], [NOTES], 0, id="local: nothing the graph describes"),
        pytest.param("commit", [CODE], [GRAPH], 1, id="commit: graph changed but not staged"),
        pytest.param("commit", [CODE, GRAPH], [], 0, id="commit: both staged"),
        pytest.param("ci", [CODE], [], 1, id="ci: a range with code alone"),
        pytest.param("ci", [CODE, GRAPH], [], 0, id="ci: a range with both"),
        pytest.param("deploy", [], [CODE], 2, id="an unknown mode is a usage error"),
    ],
)
def test_the_graph_check_reads_git_not_stdin(
    repo: Path, mode: str, staged: list[str], unstaged: list[str], code: int
) -> None:
    for path in [*staged, *unstaged]:
        (repo / path).write_text("second\n")
    if staged:
        _git(repo, "add", *staged)
    args = [mode]
    if mode == "ci":
        base = _git(repo, "rev-parse", "HEAD")
        _git(repo, "commit", "-qm", "change")
        args += [base, _git(repo, "rev-parse", "HEAD")]

    with subprocess.Popen(
        ["sh", str(SCRIPT), *args],
        cwd=repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as check:
        try:
            check.wait(timeout=10)  # stdin stays open: a script reading it never returns
        except subprocess.TimeoutExpired:
            check.kill()
            pytest.fail("the check waited on stdin")
        err = check.stderr.read() if check.stderr else ""

    assert check.returncode == code, err
