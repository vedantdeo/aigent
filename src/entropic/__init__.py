"""Entropic: a personal AI agent, built up one capability at a time over eight weeks."""

from __future__ import annotations

SCRIPTS: dict[str, str] = {
    "entropic.week01.first_call": "one raw API call, token count, cost",
    "entropic.week01.streaming": "stream text and thinking, then read the final message",
    "entropic.week01.structured_output": "Pydantic schema in, validated object out",
    "entropic.week01.tool_loop": "the agent loop written by hand",
    "entropic.week01.chat": "multi-turn CLI chat with running cost",
}


def main() -> None:
    print("Talk to the agent:  uv run entropic\nRun a step:         uv run python -m <module>\n")
    for module, what in SCRIPTS.items():
        print(f"  {module:<38} {what}")
