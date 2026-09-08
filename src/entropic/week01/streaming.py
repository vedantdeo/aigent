"""Week 1, step 2: streaming.

    uv run python -m entropic.week01.streaming

What to notice:
  - Tokens arrive as `content_block_delta` events; thinking and text are separate block types.
  - `get_final_message()` gives you the same Message object a non-streaming call returns, so usage
    and stop_reason are still there.
  - Streaming is also the fix for HTTP timeouts on long outputs. Default to it in real code.
"""

from __future__ import annotations

from anthropic.types import MessageParam

from entropic.config import MODEL, describe_usage, get_client

PROMPT = (
    "Compare BM25 and dense-embedding retrieval for a question-answering system over 10,000 PDFs. "
    "Give me the two failure modes of each, then a one-paragraph recommendation."
)


def main() -> None:
    client = get_client()
    messages: list[MessageParam] = [{"role": "user", "content": PROMPT}]

    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive", "display": "summarized"},
        messages=messages,
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    print("\n--- thinking (summarized) ---")
                elif event.content_block.type == "text":
                    print("\n--- answer ---")
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(event.delta.thinking, end="", flush=True)
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)

        final = stream.get_final_message()

    print("\n")
    print(f"stop_reason={final.stop_reason}")
    print(describe_usage(MODEL, final.usage))


if __name__ == "__main__":
    main()
