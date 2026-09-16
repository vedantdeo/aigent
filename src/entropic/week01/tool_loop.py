"""The agent loop, written by hand. What every framework wraps.

Four rules matter: the assistant turn is echoed back verbatim, every tool result for one turn goes
back in one user message, a tool error is a `tool_result` with `is_error` rather than an exception,
and the loop is capped.
"""

from __future__ import annotations

import sys

import anthropic
from anthropic.types import MessageParam, ToolResultBlockParam

from entropic.config import MAX_AGENT_TURNS as MAX_TURNS
from entropic.config import MAX_TOKENS_TOOL_LOOP as MAX_TOKENS
from entropic.config import MAX_USD_PER_RUN, MODEL, get_client
from entropic.pricing import Budget, check_request, describe_usage
from entropic.tools import ALL_TOOLS, execute_tool

SYSTEM = (
    "You are Entropic, a careful personal assistant with tools. Use the calculator for all "
    "arithmetic. Use current_time when the answer depends on today's date. Show your final answer "
    "plainly."
)


def run(task: str, client: anthropic.Anthropic | None = None) -> str:
    """Drive the loop for one task. `client` is injectable so tests can script the responses."""
    if client is None:
        client = get_client()
    messages: list[MessageParam] = [{"role": "user", "content": task}]
    budget = Budget(limit_usd=MAX_USD_PER_RUN)

    for turn in range(1, MAX_TURNS + 1):
        check_request(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            system=SYSTEM,
            tools=ALL_TOOLS,
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=ALL_TOOLS,
            messages=messages,
        )
        budget.add(MODEL, response.usage)
        usage_line = describe_usage(MODEL, response.usage)
        print(f"turn {turn}: stop_reason={response.stop_reason}  {usage_line}")

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            print(f"\ntotal cost: ${budget.spent_usd:.5f}")
            return final_text

        # Echo the assistant turn back exactly, tool_use blocks included.
        messages.append({"role": "assistant", "content": response.content})

        results: list[ToolResultBlockParam] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if not isinstance(block.input, dict):
                content, is_error = "Error: tool input was not an object", True
            else:
                content, is_error = execute_tool(block.name, block.input)
            print(
                f"   -> {block.name}({block.input}) = {content!r}{'  [error]' if is_error else ''}"
            )
            result: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": block.id,
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
            'or  uv run python -m entropic.week01.tool_loop "<task>"'
        )
    print(f"task: {task}\n")
    answer = run(task)
    print("\n--- answer ---")
    print(answer)


if __name__ == "__main__":
    main()
