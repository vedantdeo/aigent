"""Nine failure modes provoked for real, tabulated into `docs/failure-modes.md`.

Free to run by construction: a request rejected with a 4xx never reaches the model, and the last two
rows never touch the network. A successful structured call is the only thing that could make this
spend, and a test greps this file to be sure there is none.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from anthropic.types import MessageParam, Usage

from entropic.config import MODEL, get_client
from entropic.pricing import Budget, assert_request_within_budget, check_request

HELLO: list[MessageParam] = [{"role": "user", "content": "hi"}]


@dataclass(frozen=True)
class Failure:
    """One way this codebase breaks, and what it looks like when it does."""

    name: str
    when: str
    provoke: Callable[[anthropic.Anthropic], str]


def _message(exc: Exception) -> str:
    """The sentence a human needs. An SDK error stringifies to the whole JSON body, which buries
    the one field worth reading; `body` carries it parsed."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return str(exc)


def _raises(call: Callable[[], object]) -> str:
    """Run something that should fail, and report how. A success here is itself the finding."""
    try:
        call()
    except Exception as exc:  # noqa: BLE001 — cataloguing what comes out is the whole job
        detail = " ".join(_message(exc).split())
        return f"`{type(exc).__name__}` — {textwrap.shorten(detail, 150, placeholder=' …')}"
    return "**no error** — the call succeeded"


def _bad_model(client: anthropic.Anthropic) -> str:
    return _raises(
        lambda: client.messages.create(model="claude-opus-4-9", max_tokens=8, messages=HELLO)
    )


def _bad_key(_: anthropic.Anthropic) -> str:
    stranger = anthropic.Anthropic(api_key="sk-ant-not-a-real-key")
    return _raises(lambda: stranger.messages.create(model=MODEL, max_tokens=8, messages=HELLO))


def _deprecated_sampling(client: anthropic.Anthropic) -> str:
    return _raises(
        lambda: client.messages.create(
            model=MODEL, max_tokens=8, messages=HELLO, extra_body={"temperature": 0.0}
        )
    )


def _unknown_field(client: anthropic.Anthropic) -> str:
    return _raises(
        lambda: client.messages.create(
            model=MODEL, max_tokens=8, messages=HELLO, extra_body={"nonsense_field": 1}
        )
    )


def _max_tokens_too_high(client: anthropic.Anthropic) -> str:
    return _raises(
        lambda: client.messages.create(model=MODEL, max_tokens=10_000_000, messages=HELLO)
    )


def _empty_messages(client: anthropic.Anthropic) -> str:
    return _raises(lambda: client.messages.create(model=MODEL, max_tokens=8, messages=[]))


def _context_too_long(client: anthropic.Anthropic) -> str:
    """Our guard, not theirs: `check_request` prices it before the API ever sees it."""
    huge: list[MessageParam] = [{"role": "user", "content": "token " * 300_000}]
    return _raises(lambda: check_request(client, model=MODEL, max_tokens=1024, messages=huge))


def _per_request_ceiling(_: anthropic.Anthropic) -> str:
    return _raises(
        lambda: assert_request_within_budget(MODEL, input_tokens=2_000_000, max_tokens=8)
    )


def _per_run_ceiling(_: anthropic.Anthropic) -> str:
    budget = Budget(limit_usd=0.01)
    usage = Usage(input_tokens=100_000, output_tokens=100_000)
    return _raises(lambda: budget.add(MODEL, usage))


FAILURES: tuple[Failure, ...] = (
    Failure("bad-model", "a model name that does not exist", _bad_model),
    Failure("bad-key", "credentials that are not ours", _bad_key),
    Failure("deprecated-sampling", "`temperature` on a Claude 5 model", _deprecated_sampling),
    Failure("unknown-field", "a field the API has never heard of", _unknown_field),
    Failure("max-tokens-too-high", "an output cap above the model's limit", _max_tokens_too_high),
    Failure("empty-messages", "a request with nothing to answer", _empty_messages),
    Failure("context-too-long", "an input larger than the context window", _context_too_long),
    Failure("per-request-ceiling", "one call worth more than the ceiling", _per_request_ceiling),
    Failure("per-run-ceiling", "a run that spends past its budget", _per_run_ceiling),
)


def main() -> None:
    client = get_client()
    print("# Failure modes\n")
    print("What this codebase does when things go wrong, provoked rather than described.")
    print(
        "Regenerate with `uv run python -m entropic.primitives.failures > docs/failure-modes.md`."
    )
    print(
        "Every row is free: a rejected request is never billed, and the last two "
        "never leave the machine.\n"
    )
    print("| failure | provoked by | what you get |")
    print("|---|---|---|")
    for failure in FAILURES:
        print(f"| `{failure.name}` | {failure.when} | {failure.provoke(client)} |")


if __name__ == "__main__":
    main()
