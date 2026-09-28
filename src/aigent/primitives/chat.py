"""Multi-turn conversation with a running cost meter.

Keeps the whole history in memory and re-sends it each turn, so cost grows quadratically — the meter
is there to make that visible.
"""

from __future__ import annotations

from typing import Literal, cast, get_args

from aigent.config import MAX_TOKENS_CHAT as MAX_TOKENS
from aigent.config import MAX_USD_PER_RUN, MODEL
from aigent.errors import BudgetExceeded
from aigent.llm import Llm, Request
from aigent.messages import Msg
from aigent.pricing import Budget

Effort = Literal["low", "medium", "high", "xhigh"]
EFFORTS: tuple[str, ...] = get_args(Effort)

SYSTEM = (
    "You are aigent, a direct and technically precise personal assistant. "
    "Prefer short answers with concrete examples."
)


def main() -> None:
    llm = Llm(budget=Budget(limit_usd=MAX_USD_PER_RUN))
    history: list[Msg] = []
    effort: Effort = "medium"

    print(f"aigent  model={MODEL}  effort={effort}   (/effort, /reset, /quit)\n")
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
        request = Request(
            "turn", history, MAX_TOKENS, system=SYSTEM, output_config={"effort": effort}
        )
        try:
            with llm.stream(request) as stream:
                print("aigent> ", end="", flush=True)
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                final = stream.final()
        except BudgetExceeded as exc:
            history.pop()
            print(f"[not sent] {exc}\n   /reset clears the history.\n")
            continue

        history.append({"role": "assistant", "content": final.blocks})
        if llm.budget.tripped is not None:
            print(f"\n[session over] {llm.budget.tripped}")
            break
        print(
            f"\n   [in={final.usage.input_tokens} out={final.usage.output_tokens} "
            f"turn=${llm.trace[-1].usd:.5f} session=${llm.spent_usd:.5f}]\n"
        )

    print(f"session cost: ${llm.spent_usd:.5f}")


if __name__ == "__main__":
    main()
