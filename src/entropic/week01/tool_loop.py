"""Week 1, step 4: the agent loop, written by hand.

    uv run python -m entropic.week01.tool_loop

This is the whole secret. Every agent framework is a wrapper around this loop:

    while True:
        response = model(messages, tools)
        if no tool calls: break
        run the tools, append the results, continue

What to notice:
  - The assistant turn (with its tool_use blocks) goes back into history verbatim.
  - All tool results for one turn go back in ONE user message. Splitting them teaches the model to
    stop making parallel calls.
  - Errors go back as tool_result with is_error=True, not as exceptions. The model reads them.
  - A loop with no iteration cap is a cost bug waiting to happen.
"""

from __future__ import annotations

from anthropic.types import MessageParam, ToolResultBlockParam

from entropic.config import MODEL, describe_usage, get_client, usage_cost
from entropic.tools import ALL_TOOLS, execute_tool

SYSTEM = (
    "You are Entropic, a careful personal assistant with tools. Use the calculator for all "
    "arithmetic. Use current_time when the answer depends on today's date. Show your final answer "
    "plainly."
)
TASK = (
    "If I invest 250,000 rupees today at 11.5% compounded annually, what is it worth after "
    "7 years? Also tell me the current UTC time, and how many hours until midnight UTC."
)
MAX_TURNS = 8


def run(task: str) -> str:
    client = get_client()
    messages: list[MessageParam] = [{"role": "user", "content": task}]
    total_cost = 0.0

    for turn in range(1, MAX_TURNS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=ALL_TOOLS,
            messages=messages,
        )
        total_cost += usage_cost(MODEL, response.usage)
        usage_line = describe_usage(MODEL, response.usage)
        print(f"turn {turn}: stop_reason={response.stop_reason}  {usage_line}")

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            print(f"\ntotal cost: ${total_cost:.5f}")
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


def main() -> None:
    print(f"task: {TASK}\n")
    answer = run(TASK)
    print("\n--- answer ---")
    print(answer)


if __name__ == "__main__":
    main()
