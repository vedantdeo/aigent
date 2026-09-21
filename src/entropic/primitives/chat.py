"""Multi-turn conversation with a running cost meter.

Keeps the whole history in memory and re-sends it each turn, so cost grows quadratically — the meter
is there to make that visible.
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
                client,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=history,
                system=SYSTEM,
                budget=budget,
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
