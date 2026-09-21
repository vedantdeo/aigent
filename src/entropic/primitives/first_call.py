"""One non-streaming call, with its token count and cost."""

from __future__ import annotations

from entropic.config import MAX_TOKENS_FIRST_CALL as MAX_TOKENS
from entropic.config import MODEL
from entropic.llm import Llm, Request
from entropic.pricing import describe_usage

SYSTEM = "You are Entropic, an engineer's personal assistant. Answer in at most three sentences."
QUESTION = "What is a KV cache in a transformer decoder, and why does it speed up generation?"


def main() -> None:
    llm = Llm()
    request = Request.ask("first-call", SYSTEM, QUESTION, MAX_TOKENS)

    print(f"pre-flight input tokens: {llm.count(request)}\n")
    response = llm.create(request)

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
