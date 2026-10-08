"""Tool implementations, their JSON schemas, and a name->function dispatcher.

Imports nothing from this package except `tools_config`, which is what lets the pair be lifted into
an SDK tool runner or a LangGraph node unchanged.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from anthropic.types import ToolParam, WebSearchTool20260318Param

from aigent.tools_config import (
    MAX_EXPONENT,
    MAX_FILE_READ_CHARS,
    MAX_FILE_WRITE_CHARS,
    MAX_WEB_SEARCHES,
    SANDBOX,
)

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


def _in_sandbox(file_path: str, sandbox: Path) -> Path:
    """`file_path` resolved under `sandbox`, refusing one that escapes it or names a directory."""
    resolved = (sandbox / file_path).resolve()
    if not resolved.is_relative_to(sandbox.resolve()):
        raise ValueError(f"path escapes the workspace: {file_path}")
    if resolved.is_dir():
        raise ValueError(f"{file_path or '.'} is a directory, not a file")
    return resolved


def read_file(file_path: str, sandbox: Path = SANDBOX) -> str:
    """Read the contents of a file and return it as a string."""
    try:
        resolved = _in_sandbox(file_path, sandbox)
        with open(resolved, encoding="utf-8", errors="replace") as f:
            r = f.read(MAX_FILE_READ_CHARS + 1)  # one extra character tells us whether it was cut
        if len(r) > MAX_FILE_READ_CHARS:
            r = r[:MAX_FILE_READ_CHARS] + f"\n[truncated at {MAX_FILE_READ_CHARS} characters]"
        return r
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {file_path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {file_path}: {exc}") from exc


def write_file(file_path: str, content: str, sandbox: Path = SANDBOX) -> str:
    """Create a new file, and any missing parent directories, inside the sandbox.
    Never overwrites: the sandbox holds tracked eval fixtures."""
    if len(content) > MAX_FILE_WRITE_CHARS:
        raise ValueError(f"content is {len(content)} characters; the cap is {MAX_FILE_WRITE_CHARS}")
    try:
        resolved = _in_sandbox(file_path, sandbox)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" refuses an existing file atomically, where a check-then-write would race
            with open(resolved, "x", encoding="utf-8") as f:
                f.write(content)
        except FileExistsError as exc:
            raise ValueError(f"{file_path} already exists; pick a new name") from exc
    except OSError as exc:
        raise ValueError(f"Could not write {file_path}: {exc.strerror}") from exc
    return f"Wrote {len(content)} characters to {file_path}"


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

WRITE_FILE_TOOL: ToolParam = {
    "name": "write_file",
    "description": (
        "Create a new text file in the sandbox directory, with any missing parent directories. "
        "The file path is relative to the sandbox. An existing file is never overwritten: pick a "
        f"new name instead. The content can be at most {MAX_FILE_WRITE_CHARS} characters."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path of the new file.",
            },
            "content": {
                "type": "string",
                "description": "The text to write.",
            },
        },
        "required": ["file_path", "content"],
        "additionalProperties": False,
    },
    "strict": True,
}

# What the agent is offered. `write_file` is served over MCP only, so the agent's evals still hold.
ALL_TOOLS: list[ToolParam] = [CALCULATOR_TOOL, TIME_TOOL, READ_FILE_TOOL]


# Tools whose output is text someone else wrote; a model only ever sees it fenced.
UNTRUSTED_TOOLS = frozenset({"read_file"})
# Tools that change the sandbox; the dispatcher runs one only for a caller that opts in.
WRITE_TOOLS = frozenset({"write_file"})
_FENCE_TAG = re.compile(r"<(/?)untrusted", re.IGNORECASE)


def fence(source: str, content: str) -> str:
    """`content` inside an `<untrusted>` element no tag in it can close."""
    escaped = _FENCE_TAG.sub(lambda tag: f"&lt;{tag[1]}untrusted", content)
    return f'<untrusted source="{source}">\n{escaped}\n</untrusted>'


def execute_tool(
    name: str, tool_input: dict[str, object], *, sandbox: Path = SANDBOX, writes: bool = False
) -> tuple[str, bool]:
    """Dispatch by name, returning `(content, is_error)`, with an `UNTRUSTED_TOOLS` result fenced
    and a `WRITE_TOOLS` call refused unless `writes`. Never raises: the model can recover from an
    error message, not from a traceback."""
    if name in WRITE_TOOLS and not writes:
        return f"Error: {name} is not available here", True
    content, is_error = _dispatch(name, tool_input, sandbox)
    if name in UNTRUSTED_TOOLS and not is_error:
        return fence(name, content), is_error
    return content, is_error


def _dispatch(name: str, tool_input: dict[str, object], sandbox: Path) -> tuple[str, bool]:
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
            return read_file(file_path, sandbox), False

        if name == "write_file":
            file_path, content = tool_input.get("file_path"), tool_input.get("content")
            if not isinstance(file_path, str):
                return "Error: 'file_path' must be a string", True
            if not isinstance(content, str):
                return "Error: 'content' must be a string", True
            return write_file(file_path, content, sandbox), False

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
