"""What a call costs, and whether you are allowed to make it.

Every script in this repo prints what it spent. Get in the habit now; you will be asked about cost
in every interview.

Two guards, at two scales. `check_request` runs *before* a call and refuses to send one whose worst
case is over the per-request ceiling — the count itself is free, so the guard costs nothing.
`Budget` accumulates across a run and reacts *after* the call that crosses the line, which is the
only honest moment: you cannot know what a call cost until it is done.

This module is the only place that knows a price. `config` holds the ceilings as settings; nothing
else prices anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import anthropic
from anthropic import Omit, omit
from anthropic.types import MessageParam, TextBlockParam, ToolParam, Usage

from entropic.config import MAX_USD_PER_REQUEST, MAX_USD_PER_RUN


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25

    @property
    def cache_read(self) -> float:
        return self.input * 0.10


PRICES: dict[str, Price] = {
    "claude-opus-5": Price(input=5.0, output=25.0),
    "claude-sonnet-5": Price(input=2.0, output=10.0),
    "claude-haiku-4-5": Price(input=1.0, output=5.0),
    "claude-fable-5-1": Price(input=10.0, output=50.0),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Dollar cost of one request. Unknown models cost nothing rather than crashing the script."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    per_token = 1e-6
    return per_token * (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_write_tokens * price.cache_write
        + cache_read_tokens * price.cache_read
    )


def usage_cost(model: str, usage: Usage) -> float:
    return cost_usd(
        model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
    )


def describe_usage(model: str, usage: Usage) -> str:
    """One line you can paste into LOG.md."""
    return (
        f"[{model}] in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_write={usage.cache_creation_input_tokens or 0} "
        f"cache_read={usage.cache_read_input_tokens or 0} "
        f"cost=${usage_cost(model, usage):.5f}"
    )


class BudgetExceeded(RuntimeError):
    """Raised before money is spent (per request) or right after a ceiling is crossed (per run)."""


def worst_case_usd(model: str, input_tokens: int, max_tokens: int) -> float:
    """The most one request can cost: all input at the input price, the full output cap at the
    output price. Thinking counts against max_tokens, so this really is the ceiling."""
    return cost_usd(model, input_tokens=input_tokens, output_tokens=max_tokens)


def estimate_eval_usd(
    model: str,
    n_cases: int,
    input_tokens_per_case: int,
    max_tokens: int,
    calls_per_case: int = 1,
    cached_tokens: int = 0,
) -> float:
    """Worst case for a whole eval run, before the first row goes out.

    `check_request` cannot see this coming: no single row of an eval is expensive, and fifty cheap
    rows still add up to a number worth knowing in advance. Raise `calls_per_case` when a row costs
    more than one call — an LLM-as-judge grader makes it two.

    `cached_tokens` is the prefix sent behind a cache breakpoint: written once at a 25% premium,
    then read at a tenth of the input price. Leave it zero and this is the plain arithmetic it has
    always been. An estimator that cannot model the cache would refuse a cached run for costing
    what the uncached one costs, which is the opposite of useful.
    """
    calls = n_cases * calls_per_case
    if not calls:
        return 0.0
    fresh = max(input_tokens_per_case - cached_tokens, 0)
    first = cost_usd(model, fresh, max_tokens, cache_write_tokens=cached_tokens)
    rest = cost_usd(model, fresh, max_tokens, cache_read_tokens=cached_tokens)
    return first + (calls - 1) * rest


def assert_request_within_budget(
    model: str, input_tokens: int, max_tokens: int, limit_usd: float = MAX_USD_PER_REQUEST
) -> float:
    """Pure check, no network. Returns the worst-case cost, or raises BudgetExceeded."""
    worst = worst_case_usd(model, input_tokens, max_tokens)
    if worst > limit_usd:
        raise BudgetExceeded(
            f"request could cost up to ${worst:.4f} ({input_tokens} input tokens plus "
            f"max_tokens={max_tokens} on {model}), above the per-request ceiling of "
            f"${limit_usd:.4f}. Trim the input, lower max_tokens, or raise "
            "ENTROPIC_MAX_USD_PER_REQUEST in .env."
        )
    return worst


def check_request(
    client: anthropic.Anthropic,
    *,
    model: str,
    max_tokens: int,
    messages: Sequence[MessageParam],
    system: str | Sequence[TextBlockParam] | Omit = omit,
    tools: Sequence[ToolParam] | Omit = omit,
) -> int:
    """Count the input tokens (a free call), then enforce the per-request ceiling.

    Pass exactly what the real request will send, so the count is the real count — `system` takes
    the block form too, which is how a cache breakpoint travels. Returns the input token count so
    callers can print it.
    """
    count = client.messages.count_tokens(model=model, messages=messages, system=system, tools=tools)
    assert_request_within_budget(model, count.input_tokens, max_tokens)
    return count.input_tokens


@dataclass
class Budget:
    """Running spend for one run: a tool loop, a chat session, an eval. Trips after the call that
    crosses the ceiling, so spent_usd always reflects what was actually billed.

    Two ways to bill, because two kinds of run want different things when the money runs out.
    `add` raises, which is right for a tool loop or a chat session: there is a person waiting and
    nothing useful left to do. `charge` records the trip on the budget and returns, which is right
    for a batch that should stop cleanly and still report the rows it paid for.

    `scope` only shapes the message, but it earns its place there: the guard's promise is that it
    names the knob to turn, and an eval that reports the per-run env var sends you to the wrong line
    of .env.
    """

    limit_usd: float = MAX_USD_PER_RUN
    spent_usd: float = 0.0
    scope: str = "run"
    tripped: str | None = None

    def charge(self, model: str, usage: Usage) -> float:
        """Bill one call and return what it added. Never raises; sets `tripped` when over."""
        charged = usage_cost(model, usage)
        self.spent_usd += charged
        if self.spent_usd > self.limit_usd:
            self.tripped = (
                f"{self.scope} has spent ${self.spent_usd:.4f}, above the per-{self.scope} "
                f"ceiling of ${self.limit_usd:.4f}. Raise "
                f"ENTROPIC_MAX_USD_PER_{self.scope.upper()} in .env if this was intended."
            )
        return charged

    def add(self, model: str, usage: Usage) -> float:
        """Bill one call and raise once the ceiling is crossed. Returns the running total."""
        self.charge(model, usage)
        if self.tripped is not None:
            raise BudgetExceeded(self.tripped)
        return self.spent_usd
