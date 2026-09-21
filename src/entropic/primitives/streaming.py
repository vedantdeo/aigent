"""A streamed call, showing thinking and text blocks arriving separately.

Usage totals only arrive at the end of the stream, so the cost line comes after the text.
"""

from __future__ import annotations

from anthropic.types import MessageParam

from entropic.config import MAX_TOKENS_STREAMING as MAX_TOKENS
from entropic.config import MODEL
from entropic.llm import Llm, Request
from entropic.pricing import describe_usage

PROMPT = (
    "Compare BM25 and dense-embedding retrieval for a question-answering system over 10,000 PDFs. "
    "Give me the two failure modes of each, then a one-paragraph recommendation."
)


def main() -> None:
    messages: list[MessageParam] = [{"role": "user", "content": PROMPT}]
    request = Request(
        "stream",
        messages,
        MAX_TOKENS,
        thinking={"type": "adaptive", "display": "summarized"},
    )

    with Llm().stream(request) as stream:
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
