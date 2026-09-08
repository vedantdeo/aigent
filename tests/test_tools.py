from __future__ import annotations

import pytest

from entropic.tools import calculate, execute_tool


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
