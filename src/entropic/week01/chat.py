"""Week 1, step 5: a multi-turn CLI chat with a running cost meter.

    uv run python -m entropic.week01.chat

Commands:  /effort low|medium|high|xhigh   /reset   /quit

What to notice:
  - History grows every turn, and you pay for all of it every turn. Watch input tokens climb.
  - Effort changes cost and latency far more than most prompt tweaks. Try the same question at low
    and high and compare.
  - The system prompt is identical every call. That is what makes it cacheable (Week 2).
  - Two guards. A turn that could exceed the per-request ceiling is not sent (use /reset). The
    session ends when the per-run ceiling is crossed.
"""

from __future__ import annotations

from typing import Literal, cast, get_args

from anthropic.types import MessageParam

from entropic.config import MAX_TOKENS_CHAT as MAX_TOKENS
from entropic.config import MAX_USD_PER_RUN, MODEL, get_client
from entropic.pricing import Budget, BudgetExceeded, check_request, usage_cost

Effort = Literal["low", "medium", "high", "xhigh"]
EFFORTS: tuple[str, ...] = get_args(Effort)

SYSTEM = (
    "You are Entropic, a direct and technically precise personal assistant. "
    "Prefer short answers with concrete examples."
)


def main() -> None:
    client = get_client()
    history: list[MessageParam] = []
    effort: Effort = "medium"
    budget = Budget(limit_usd=MAX_USD_PER_RUN)

    print(f"entropic  model={MODEL}  effort={effort}   (/effort, /reset, /quit)\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/reset":
            history.clear()
            print("history cleared\n")
            continue
        if user_text.startswith("/effort"):
            parts = user_text.split()
            if len(parts) == 2 and parts[1] in EFFORTS:
                effort = cast(Effort, parts[1])
                print(f"effort={effort}\n")
            else:
                print(f"usage: /effort {'|'.join(EFFORTS)}\n")
            continue

        history.append({"role": "user", "content": user_text})
        try:
            check_request(
                client, model=MODEL, max_tokens=MAX_TOKENS, messages=history, system=SYSTEM
            )
        except BudgetExceeded as exc:
            history.pop()
            print(f"[not sent] {exc}\n   /reset clears the history.\n")
            continue
        print("entropic> ", end="", flush=True)
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            output_config={"effort": effort},
            messages=history,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()

        history.append({"role": "assistant", "content": final.content})
        turn_cost = usage_cost(MODEL, final.usage)
        try:
            budget.add(MODEL, final.usage)
        except BudgetExceeded as exc:
            print(f"\n[session over] {exc}")
            break
        print(
            f"\n   [in={final.usage.input_tokens} out={final.usage.output_tokens} "
            f"turn=${turn_cost:.5f} session=${budget.spent_usd:.5f}]\n"
        )

    print(f"session cost: ${budget.spent_usd:.5f}")


if __name__ == "__main__":
    main()
