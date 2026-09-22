"""Tool implementations, their JSON schemas, and a name->function dispatcher.

Imports nothing from this package except `tools_config`, which is what lets the pair be lifted into
an SDK tool runner or a LangGraph node unchanged.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from anthropic.types import ToolParam, WebSearchTool20260318Param

from entropic.tools_config import MAX_EXPONENT, MAX_FILE_READ_CHARS, MAX_WEB_SEARCHES, SANDBOX

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


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
            if isinstance(op, ast.Pow) and abs(rhs) > MAX_EXPONENT:
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
            r = f.read(MAX_FILE_READ_CHARS + 1)  # one extra character tells us whether it was cut
        if len(r) > MAX_FILE_READ_CHARS:
            r = r[:MAX_FILE_READ_CHARS] + f"\n[truncated at {MAX_FILE_READ_CHARS} characters]"
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
        f"cannot be listed. The maximum file size that can be read is {MAX_FILE_READ_CHARS} "
        "characters. Files larger than this will be truncated with a cut marker at the end."
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
    """Dispatch by name, returning `(content, is_error)`. Never raises.

    The model can recover from an error message; it cannot recover from a traceback.
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


# A server tool: the API runs it, so it has a schema here and no implementation or dispatch branch.
# Called directly, so its results come back with the pages they cite.
WEB_SEARCH_TOOL: WebSearchTool20260318Param = {
    "type": "web_search_20260318",
    "name": "web_search",
    "max_uses": MAX_WEB_SEARCHES,
    "allowed_callers": ["direct"],
}
