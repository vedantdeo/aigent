from __future__ import annotations

import os
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


def test_read_file_refuses_parent_traversal(box: Path) -> None:
    with pytest.raises(ValueError, match="escapes the workspace"):
        read_file("../outside.txt", sandbox=box)


def test_read_file_refuses_absolute_paths(box: Path) -> None:
    outside = (box.parent / "outside.txt").resolve()
    with pytest.raises(ValueError, match="escapes the workspace"):
        read_file(str(outside), sandbox=box)


def test_read_file_refuses_symlinks_that_point_outside(box: Path) -> None:
    os.symlink(box.parent / "outside.txt", box / "link.txt")
    with pytest.raises(ValueError, match="escapes the workspace"):
        read_file("link.txt", sandbox=box)


def test_read_file_missing_file_is_a_flagged_error_through_the_dispatcher() -> None:
    content, is_error = execute_tool("read_file", {"file_path": "definitely-not-here.txt"})
    assert is_error is True
    assert content.startswith("Error: File not found")


def test_read_file_directory_is_a_short_error_without_the_absolute_path(box: Path) -> None:
    (box / "sub").mkdir()
    with pytest.raises(ValueError, match="^sub is a directory, not a file$"):
        read_file("sub", sandbox=box)
    with pytest.raises(ValueError, match="^\\. is a directory, not a file$"):
        read_file("", sandbox=box)


def test_read_file_truncates_long_files_with_a_marker(box: Path) -> None:
    (box / "big.txt").write_text("x" * (MAX_FILE_READ_CHARS + 500), encoding="utf-8")
    body, marker = read_file("big.txt", sandbox=box).rsplit("\n", 1)
    assert len(body) == MAX_FILE_READ_CHARS
    assert marker == f"[truncated at {MAX_FILE_READ_CHARS} characters]"


def test_read_file_at_exactly_the_cap_is_not_truncated(box: Path) -> None:
    (box / "exact.txt").write_text("y" * MAX_FILE_READ_CHARS, encoding="utf-8")
    assert read_file("exact.txt", sandbox=box) == "y" * MAX_FILE_READ_CHARS


def test_execute_tool_rejects_a_non_string_path() -> None:
    content, is_error = execute_tool("read_file", {"file_path": 42})
    assert is_error is True
    assert "must be a string" in content


# --- dispatcher ----------------------------------------------------------------------------


def test_execute_tool_error_path_is_not_an_exception() -> None:
    content, is_error = execute_tool("calculate", {"expression": "1 / 0"})
    assert is_error is True
    assert content.startswith("Error:")


def test_execute_tool_unknown_name() -> None:
    content, is_error = execute_tool("nope", {})
    assert is_error is True
    assert "nope" in content


def test_execute_tool_time() -> None:
    content, is_error = execute_tool("current_time", {})
    assert is_error is False
    assert content.endswith("+00:00")
