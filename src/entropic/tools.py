"""Tools the model can call. Kept framework-free so the same functions work in Week 1's hand-written
loop, Week 5's SDK tool runner, and Week 6's LangGraph agent.

A tool is three things: a schema the model sees, a function you run, and an error path the model can
recover from. Get all three right and most "agent bugs" disappear.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from anthropic.types import ToolParam

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_MAX_EXPONENT = 1000.0

SANDBOX = Path(__file__).resolve().parents[2] / "sandbox"
_MAX_FILE_READ = 20000  # 20k characters


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression safely. No names, no calls, no attribute access."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid expression: {expression!r}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> float:
    match node:
        case ast.Constant(value=value) if isinstance(value, int | float) and not isinstance(
            value, bool
        ):
            return float(value)
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval_node(operand)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _eval_node(operand)
        case ast.BinOp(left=left, op=op, right=right):
            fn = _BINARY_OPS.get(type(op))
            if fn is None:
                raise ValueError(f"operator not allowed: {type(op).__name__}")
            lhs, rhs = _eval_node(left), _eval_node(right)
            if isinstance(op, ast.Pow) and abs(rhs) > _MAX_EXPONENT:
                raise ValueError(f"exponent too large: {rhs}")
            if isinstance(op, ast.Div | ast.Mod) and rhs == 0:
                raise ValueError("division by zero")
            return fn(lhs, rhs)
        case _:
            raise ValueError(f"unsupported syntax: {type(node).__name__}")


def current_time() -> str:
    """ISO-8601 UTC timestamp. The model has no clock; this is the classic trivial-but-real tool."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_file(file_path: str, sandbox: Path = SANDBOX) -> str:
    """Read the contents of a file and return it as a string."""
    try:
        resolved = (sandbox / file_path).resolve()
        if not resolved.is_relative_to(sandbox.resolve()):
            raise ValueError(f"path escapes the workspace: {file_path}")
        if resolved.is_dir():
            raise ValueError(f"{file_path or '.'} is a directory, not a file")
        with open(resolved, encoding="utf-8", errors="replace") as f:
            r = f.read(_MAX_FILE_READ + 1)  # one extra character tells us whether it was cut
        if len(r) > _MAX_FILE_READ:
            r = r[:_MAX_FILE_READ] + f"\n[truncated at {_MAX_FILE_READ} characters]"
        return r
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {file_path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {file_path}: {exc}") from exc


CALCULATOR_TOOL: ToolParam = {
    "name": "calculate",
    "description": (
        "Evaluate an arithmetic expression and return the numeric result. Supports + - * / ** % "
        "and parentheses. Use this for any arithmetic instead of computing in your head."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(1200 * 1.07) ** 5'",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    "strict": True,
}

TIME_TOOL: ToolParam = {
    "name": "current_time",
    "description": "Return the current date and time in UTC as an ISO-8601 string.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}

READ_FILE_TOOL: ToolParam = {
    "name": "read_file",
    "description": (
        "Read the contents of a file and return it as a string. The file path is relative to the "
        "sandbox directory. Files outside the sandbox are not accessible. Contents of the sandbox "
        f"cannot be listed. The maximum file size that can be read is {_MAX_FILE_READ} characters. "
        "Files larger than this will be truncated with a cut marker at the end."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read.",
            }
        },
        "required": ["file_path"],
        "additionalProperties": False,
    },
    "strict": True,
}

ALL_TOOLS: list[ToolParam] = [CALCULATOR_TOOL, TIME_TOOL, READ_FILE_TOOL]


def execute_tool(name: str, tool_input: dict[str, object]) -> tuple[str, bool]:
    """Run a tool by name. Returns (content, is_error).

    Errors go back to the model as text with is_error=True, never as a raised exception. The model
    can read the message and try again; a crashed loop cannot.
    """
    try:
        if name == "calculate":
            expression = tool_input.get("expression")
            if not isinstance(expression, str):
                return "Error: 'expression' must be a string", True
            return str(calculate(expression)), False

        if name == "current_time":
            return current_time(), False

        if name == "read_file":
            file_path = tool_input.get("file_path")
            if not isinstance(file_path, str):
                return "Error: 'file_path' must be a string", True
            return read_file(file_path), False

        return f"Error: unknown tool {name!r}", True
    except Exception as exc:
        return f"Error: {exc}", True
