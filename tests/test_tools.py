from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from entropic.tools import calculate, execute_tool, read_file
from entropic.tools_config import MAX_FILE_READ_CHARS


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", 3.0),
        ("(1200 * 1.07) ** 5", (1200 * 1.07) ** 5),
        ("-3 * 4", -12.0),
        ("10 % 3", 1.0),
        ("2 ** 10", 1024.0),
    ],
)
def test_calculate(expression: str, expected: float) -> None:
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "a + 1", "1 / 0", "2 ** 100000", "(1, 2)", "True + 1", "1 +"],
)
def test_calculate_rejects(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


# --- read_file -----------------------------------------------------------------------------


@pytest.fixture
def box(tmp_path: Path) -> Path:
    """A throwaway sandbox with one file inside it and one file outside it."""
    sandbox = tmp_path / "box"
    sandbox.mkdir()
    (sandbox / "notes.txt").write_text("hello from the sandbox\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret\n", encoding="utf-8")
    return sandbox


def test_read_file_reads_inside_the_sandbox(box: Path) -> None:
    assert read_file("notes.txt", sandbox=box) == "hello from the sandbox\n"


def _symlink_pointing_out(box: Path) -> str:
    os.symlink(box.parent / "outside.txt", box / "link.txt")
    return "link.txt"


@pytest.mark.parametrize(
    "escape",
    [
        pytest.param(lambda box: "../outside.txt", id="parent traversal"),
        pytest.param(
            lambda box: str((box.parent / "outside.txt").resolve()), id="an absolute path"
        ),
        pytest.param(_symlink_pointing_out, id="a symlink pointing out"),
    ],
)
def test_read_file_refuses_a_path_that_leaves_the_sandbox(
    box: Path, escape: Callable[[Path], str]
) -> None:
    with pytest.raises(ValueError, match="escapes the workspace"):
        read_file(escape(box), sandbox=box)


def test_read_file_directory_is_a_short_error_without_the_absolute_path(box: Path) -> None:
    (box / "sub").mkdir()
    with pytest.raises(ValueError, match="^sub is a directory, not a file$"):
        read_file("sub", sandbox=box)
    with pytest.raises(ValueError, match="^\\. is a directory, not a file$"):
        read_file("", sandbox=box)


@pytest.mark.parametrize(
    ("size", "truncated"),
    [
        pytest.param(MAX_FILE_READ_CHARS, False, id="exactly at the cap"),
        pytest.param(MAX_FILE_READ_CHARS + 500, True, id="over the cap"),
    ],
)
def test_read_file_truncates_only_past_the_cap(box: Path, size: int, truncated: bool) -> None:
    (box / "big.txt").write_text("x" * size, encoding="utf-8")
    content = read_file("big.txt", sandbox=box)

    marker = f"\n[truncated at {MAX_FILE_READ_CHARS} characters]"
    assert content.endswith(marker) is truncated
    assert len(content.removesuffix(marker)) == min(size, MAX_FILE_READ_CHARS)


# --- dispatcher ----------------------------------------------------------------------------
# Every row is the same call with different arguments, and the contract never changes: a tuple,
# never an exception, with `is_error` telling the model which kind of answer it is looking at.


@pytest.mark.parametrize(
    ("tool", "tool_input", "is_error", "expect"),
    [
        ("calculate", {"expression": "2 ** 10"}, False, "1024.0"),
        ("current_time", {}, False, "+00:00"),
        ("calculate", {"expression": "1 / 0"}, True, "Error:"),
        ("read_file", {"file_path": "definitely-not-here.txt"}, True, "Error: File not found"),
        ("read_file", {"file_path": 42}, True, "must be a string"),
        ("nope", {}, True, "nope"),
    ],
)
def test_execute_tool(
    tool: str, tool_input: dict[str, object], is_error: bool, expect: str
) -> None:
    content, flagged = execute_tool(tool, tool_input)

    assert flagged is is_error, content
    assert expect in content
