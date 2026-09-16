"""One non-streaming call, with its token count and cost."""

from __future__ import annotations

from anthropic.types import MessageParam

from entropic.config import MAX_TOKENS_FIRST_CALL as MAX_TOKENS
from entropic.config import MODEL, get_client
from entropic.pricing import check_request, describe_usage

SYSTEM = "You are Entropic, an engineer's personal assistant. Answer in at most three sentences."
QUESTION = "What is a KV cache in a transformer decoder, and why does it speed up generation?"


def main() -> None:
    client = get_client()
    messages: list[MessageParam] = [{"role": "user", "content": QUESTION}]

    input_tokens = check_request(
        client, model=MODEL, max_tokens=MAX_TOKENS, messages=messages, system=SYSTEM
    )
    print(f"pre-flight input tokens: {input_tokens}\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=messages,
    )

    if response.stop_reason == "refusal":
        details = response.stop_details
        print(f"refused: {details.category if details else 'unknown'}")
        return
    if response.stop_reason == "max_tokens":
        print("warning: output was cut off by max_tokens\n")

    for block in response.content:
        if block.type == "text":
            print(block.text)

    print()
    print(f"stop_reason={response.stop_reason}  request_id={response._request_id}")
    print(describe_usage(MODEL, response.usage))


if __name__ == "__main__":
    main()
