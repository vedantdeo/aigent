"""Token prices, cost arithmetic, and the two budget guards. The only module that knows a price.

`assert_request_within_budget` refuses one call whose worst case is over the per-request ceiling.
`Budget` bills what each call actually cost once it is done, and `Budget.admit` refuses beforehand
any call whose worst case would carry spending past the ceiling. `llm` applies both to every call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from entropic.config import MAX_USD_PER_REQUEST, MAX_USD_PER_RUN, WEB_SEARCH_RESULT_TOKENS
from entropic.errors import BudgetExceeded
from entropic.messages import Usage


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


# A local model is priced by what an hour of this machine costs, divided by the tokens an hour
# buys: about $0.06/hour (~25 W of electricity, plus a 16 GB M4 amortised over three years) over
# the throughput measured in the underhood quantization bench — 490 prefill and 50 decode tokens a
# second for a 4-bit 3B, scaled by parameter count for these two. Output costs ~10x input because
# decode is ~10x slower than prefill, which is the same asymmetry a hosted price carries.
# Estimates, not measurements, for these two models; free is not a price, and $0.00 would make
# every guard read a local run as costless.
PRICES: dict[str, Price] = {
    "claude-opus-5": Price(input=5.0, output=25.0),
    "claude-sonnet-5": Price(input=2.0, output=10.0),
    "claude-haiku-4-5": Price(input=1.0, output=5.0),
    "claude-fable-5-1": Price(input=10.0, output=50.0),
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": Price(input=0.04, output=0.42),
    "mlx-community/Qwen3-4B-Instruct-2507-6bit": Price(input=0.04, output=0.59),
    "mlx-community/Qwen3-4B-Instruct-2507-8bit": Price(input=0.04, output=0.77),
    "mlx-community/Qwen3-4B-Instruct-2507-bf16": Price(input=0.04, output=1.44),
    "mlx-community/Qwen3-8B-4bit": Price(input=0.08, output=0.71),
}

# Billed per search, on top of the tokens its results add.
PRICE_PER_WEB_SEARCH = 0.01


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    web_searches: int = 0,
) -> float:
    """Dollar cost of one request. Unknown models cost nothing rather than crashing the script."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    per_token = 1e-6
    return web_searches * PRICE_PER_WEB_SEARCH + per_token * (
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
        cache_write_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        web_searches=usage.web_searches,
    )


def describe_usage(model: str, usage: Usage) -> str:
    """One line you can paste into LOG.md."""
    return (
        f"[{model}] in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_write={usage.cache_write_tokens} "
        f"cache_read={usage.cache_read_tokens} "
        + (f"web_searches={usage.web_searches} " if usage.web_searches else "")
        + f"cost=${usage_cost(model, usage):.5f}"
    )


def worst_case_usd(
    model: str, input_tokens: int, max_tokens: int, *, cached: bool = False, web_searches: int = 0
) -> float:
    """The most one request can cost: the full output cap, and all input at the input price — or at
    the cache-write price if `cached`, since a miss writes the whole prefix. Each web search adds
    its fee and `WEB_SEARCH_RESULT_TOKENS` of results: the one part estimated, not bounded."""
    input_tokens += web_searches * WEB_SEARCH_RESULT_TOKENS
    if cached:
        return cost_usd(
            model, 0, max_tokens, cache_write_tokens=input_tokens, web_searches=web_searches
        )
    return cost_usd(model, input_tokens, max_tokens, web_searches=web_searches)


def affordable_output_tokens(
    model: str, input_tokens: int, limit_usd: float, *, cached: bool = False, web_searches: int = 0
) -> int:
    """The largest `max_tokens` whose worst case, with this input, stays within `limit_usd`."""
    price = PRICES.get(model)
    if price is None:
        return 0
    spare = limit_usd - worst_case_usd(
        model, input_tokens, 0, cached=cached, web_searches=web_searches
    )
    # One token short of the exact figure, so float rounding cannot carry it over the limit.
    return max(0, math.floor(spare / (price.output * 1e-6)) - 1)


def estimate_eval_usd(
    model: str,
    n_cases: int,
    input_tokens_per_case: int,
    max_tokens: int,
    calls_per_case: int = 1,
    cached_tokens: int = 0,
) -> float:
    """Worst case for a whole run: cases x calls-per-case x `worst_case_usd`.

    `cached_tokens` prices a cached prefix as one write plus n-1 reads. Note the sign flips at n=1:
    caching a single row costs more than not caching it.
    """
    calls = n_cases * calls_per_case
    if not calls:
        return 0.0
    fresh = max(input_tokens_per_case - cached_tokens, 0)
    first = cost_usd(model, fresh, max_tokens, cache_write_tokens=cached_tokens)
    rest = cost_usd(model, fresh, max_tokens, cache_read_tokens=cached_tokens)
    return first + (calls - 1) * rest


def assert_request_within_budget(
    model: str,
    input_tokens: int,
    max_tokens: int,
    limit_usd: float = MAX_USD_PER_REQUEST,
    *,
    cached: bool = False,
    scope: str = "request",
    web_searches: int = 0,
) -> float:
    """Pure check, no network. Returns the worst-case cost, or raises BudgetExceeded.

    A model with no price is refused too: its worst case would read as free, and pass every guard.
    """
    if model not in PRICES:
        raise BudgetExceeded(
            f"{model} has no price, so no ceiling can hold it: its calls would be admitted and "
            "billed as free. Add it to pricing.PRICES."
        )
    worst = worst_case_usd(
        model, input_tokens, max_tokens, cached=cached, web_searches=web_searches
    )
    if worst > limit_usd:
        written = " written to cache" if cached else ""
        written += f", {web_searches} web searches" if web_searches else ""
        raise BudgetExceeded(
            f"request could cost up to ${worst:.4f} ({input_tokens} input tokens{written} plus "
            f"max_tokens={max_tokens} on {model}), above the per-{scope} ceiling of "
            f"${limit_usd:.4f}. Trim the input, lower max_tokens, or raise "
            f"ENTROPIC_MAX_USD_PER_{scope.upper()} in .env."
        )
    return worst


@dataclass
class Budget:
    """Per-run accumulator. `charge` records an overrun in `tripped`; `add` also raises.

    Both bill before they trip, so `spent_usd` always includes the crossing call. `scope` picks
    which env var the message names.
    """

    limit_usd: float = MAX_USD_PER_RUN
    spent_usd: float = 0.0
    scope: str = "run"
    tripped: str | None = None
    held_usd: float = 0.0  # worst cases of calls admitted and not yet billed

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

    def admit(self, worst_usd: float, *, hold: bool = False) -> None:
        """Refuse, before anything is sent, spending whose worst case — with every call still in
        flight — would carry past the ceiling. With `hold`, keep `worst_usd` held until `release`.

        Admitting bills nothing; `charge` still bills the actual cost once a call returns.
        """
        if self.spent_usd + self.held_usd + worst_usd > self.limit_usd:
            held = f" with ${self.held_usd:.4f} held for calls in flight" if self.held_usd else ""
            raise BudgetExceeded(
                f"{self.scope} has spent ${self.spent_usd:.4f}{held}, and the next spending could "
                f"cost up to ${worst_usd:.4f}, past the per-{self.scope} ceiling of "
                f"${self.limit_usd:.4f}. Raise ENTROPIC_MAX_USD_PER_{self.scope.upper()} in .env "
                "if this was intended."
            )
        if hold:
            self.held_usd += worst_usd

    def release(self, worst_usd: float) -> None:
        """Let go of a hold, once its call is billed or has failed."""
        self.held_usd = max(0.0, self.held_usd - worst_usd)
