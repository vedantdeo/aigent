"""Week 1, step 5: a multi-turn CLI chat with a running cost meter.

    uv run python -m entropic.week01.chat

Commands:  /effort low|medium|high|xhigh   /reset   /quit

What to notice:
  - History grows every turn, and you pay for all of it every turn. Watch input tokens climb.
  - Effort changes cost and latency far more than most prompt tweaks. Try the same question at low
    and high and compare.
  - The system prompt is identical every call. That is what makes it cacheable (Week 2).
"""

from __future__ import annotations

from typing import Literal, cast, get_args

from anthropic.types import MessageParam

from entropic.config import MODEL, get_client, usage_cost

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
    session_cost = 0.0

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
        print("entropic> ", end="", flush=True)
        with client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            output_config={"effort": effort},
            messages=history,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()

        history.append({"role": "assistant", "content": final.content})
        turn_cost = usage_cost(MODEL, final.usage)
        session_cost += turn_cost
        print(
            f"\n   [in={final.usage.input_tokens} out={final.usage.output_tokens} "
            f"turn=${turn_cost:.5f} session=${session_cost:.5f}]\n"
        )

    print(f"session cost: ${session_cost:.5f}")


if __name__ == "__main__":
    main()
