"""The `entropic` command: a mode table, a lookup, an interactive menu.

`entropic <mode>` runs one of the five Week 1 primitives; bare `entropic` opens the menu.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from entropic.week01 import chat, first_call, streaming, structured_output, tool_loop


@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    blurb: str
    run: Callable[[list[str]], None]


def _run_loop(rest: list[str]) -> None:
    if not rest and sys.stdin.isatty():
        rest = [input("task> ").strip()]
    tool_loop.main(rest)


MODES: tuple[Mode, ...] = (
    Mode("call", "first call", "one raw API call, token count, cost", lambda _: first_call.main()),
    Mode(
        "stream",
        "streaming",
        "watch thinking and text arrive separately",
        lambda _: streaming.main(),
    ),
    Mode(
        "extract",
        "structured output",
        "schema in, validated object out",
        lambda _: structured_output.main(),
    ),
    Mode("loop", "tool loop", "the agent loop written by hand; takes a task", _run_loop),
    Mode(
        "chat", "chat", "multi-turn conversation with a running cost meter", lambda _: chat.main()
    ),
)


def find_mode(name: str) -> Mode | None:
    """Look a mode up by key ("chat") or by its 1-based menu number ("5")."""
    name = name.strip().lower()
    if name.isdigit():
        index = int(name) - 1
        return MODES[index] if 0 <= index < len(MODES) else None
    return next((m for m in MODES if m.key == name), None)


def menu_text() -> str:
    lines = ["entropic modes:"]
    for number, mode in enumerate(MODES, start=1):
        lines.append(f"  {number}. {mode.key:<8} {mode.title:<18} {mode.blurb}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    if args:
        mode = find_mode(args[0])
        if mode is None:
            raise SystemExit(f"unknown mode {args[0]!r}\n{menu_text()}")
        mode.run(args[1:])
        return

    print(menu_text())
    if not sys.stdin.isatty():
        raise SystemExit("no terminal attached; pick a mode: uv run entropic <mode>")
    choice = input("\nmode> ")
    mode = find_mode(choice)
    if mode is None:
        raise SystemExit(f"unknown mode {choice.strip()!r}")
    print()
    mode.run([])
