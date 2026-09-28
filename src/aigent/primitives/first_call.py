"""One non-streaming call, with its token count and cost."""

from __future__ import annotations

from typing import cast

from anthropic.types import Message

from aigent.config import MAX_TOKENS_FIRST_CALL as MAX_TOKENS
from aigent.config import MODEL
from aigent.llm import Llm, Request
from aigent.pricing import describe_usage

SYSTEM = "You are aigent, an engineer's personal assistant. Answer in at most three sentences."
QUESTION = "What is a KV cache in a transformer decoder, and why does it speed up generation?"


def main() -> None:
    llm = Llm()
    request = Request.ask("first-call", SYSTEM, QUESTION, MAX_TOKENS)

    print(f"pre-flight input tokens: {llm.count(request)}\n")
    reply = llm.create(request)
    # The refusal category and the request id are Anthropic's own; this demo is about its API.
    message = cast(Message, reply.raw)

    if reply.stop_reason == "refusal":
        details = message.stop_details
        print(f"refused: {details.category if details else 'unknown'}")
        return
    if reply.stop_reason == "max_tokens":
        print("warning: output was cut off by max_tokens\n")

    print(reply.text)

    print()
    print(f"stop_reason={reply.stop_reason}  request_id={message._request_id}")
    print(describe_usage(MODEL, reply.usage))


if __name__ == "__main__":
    main()
