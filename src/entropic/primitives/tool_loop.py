"""The agent loop, written by hand. What every framework wraps.

Four rules matter: the assistant turn is echoed back verbatim, every tool result for one turn goes
back in one user message, a tool error is a `tool_result` with `is_error` rather than an exception,
and the loop is capped.
"""

from __future__ import annotations

import sys
from typing import cast

from entropic.config import MAX_AGENT_TURNS as MAX_TURNS
from entropic.config import MAX_TOKENS_TOOL_LOOP as MAX_TOKENS
from entropic.config import MAX_USD_PER_RUN, MODEL
from entropic.errors import BudgetExceeded
from entropic.llm import Llm, Request
from entropic.messages import Block, Msg
from entropic.pricing import Budget, describe_usage
from entropic.tools import ALL_TOOLS, execute_tool

SYSTEM = (
    "You are Entropic, a careful personal assistant with tools. Use the calculator for all "
    "arithmetic. Use current_time when the answer depends on today's date. Show your final answer "
    "plainly."
)


def run(task: str, sdk: object | None = None, *, limit_usd: float = MAX_USD_PER_RUN) -> str:
    """Drive the loop for one task. `sdk` is injectable so tests can script the replies."""
    llm = Llm(sdk=sdk, budget=Budget(limit_usd=limit_usd))
    messages: list[Msg] = [{"role": "user", "content": task}]

    for turn in range(1, MAX_TURNS + 1):
        request = Request(f"turn:{turn}", messages, MAX_TOKENS, system=SYSTEM, tools=ALL_TOOLS)
        response = llm.create(request)
        if llm.budget.tripped is not None:
            raise BudgetExceeded(llm.budget.tripped)
        usage_line = describe_usage(MODEL, response.usage)
        print(f"turn {turn}: stop_reason={response.stop_reason}  {usage_line}")

        if response.stop_reason != "tool_use":
            print(f"\ntotal cost: ${llm.spent_usd:.5f}")
            return response.text

        # Echo the assistant turn back exactly, tool_use blocks included.
        messages.append({"role": "assistant", "content": response.blocks})

        results: list[Block] = []
        for block in response.blocks:
            if block.get("type") != "tool_use":
                continue
            if not isinstance(block.get("input"), dict):
                content, is_error = "Error: tool input was not an object", True
            else:
                content, is_error = execute_tool(
                    cast(str, block["name"]), cast(dict[str, object], block["input"])
                )
            name, arguments = block["name"], block["input"]
            print(f"   -> {name}({arguments}) = {content!r}{'  [error]' if is_error else ''}")
            result: dict[str, object] = {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": content,
            }
            if is_error:
                result["is_error"] = True
            results.append(result)

        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"agent did not finish within {MAX_TURNS} turns")


def main(argv: list[str] | None = None) -> None:
    task = " ".join(sys.argv[1:] if argv is None else argv).strip()
    if not task:
        raise SystemExit(
            'usage: uv run entropic loop "<task>"  '
            'or  uv run python -m entropic.primitives.tool_loop "<task>"'
        )
    print(f"task: {task}\n")
    answer = run(task)
    print("\n--- answer ---")
    print(answer)


if __name__ == "__main__":
    main()
