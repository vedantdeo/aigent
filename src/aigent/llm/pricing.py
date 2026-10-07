"""Token prices, cost arithmetic, and the two budget guards. The only module that knows a price.

`assert_request_within_budget` refuses one call whose worst case is over the per-request ceiling.
`Budget` bills what each call actually cost once it is done, and `Budget.admit` refuses beforehand
any call whose worst case would carry spending past the ceiling. `llm` applies both to every call.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from aigent.config import MAX_USD_PER_REQUEST, MAX_USD_PER_RUN, WEB_SEARCH_RESULT_TOKENS
from aigent.errors import BudgetExceeded
from aigent.llm.messages import CacheTtl, Usage


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cached_input: float | None = None  # a provider's own cache-read price, where it sets one

    @property
    def cache_write(self) -> float:
        return self.input * 1.25

    @property
    def cache_write_1h(self) -> float:
        return self.input * 2.0

    @property
    def cache_read(self) -> float:
        return self.input * 0.10 if self.cached_input is None else self.cached_input


# A local model costs the electricity an hour of decoding draws (~25 W at ₹8/kWh, $0.0021/hour)
# over the tokens that hour buys; the machine is owned either way. Estimates, and never $0.00.
PRICES: dict[str, Price] = {
    "claude-opus-5": Price(input=5.0, output=25.0),
    "claude-sonnet-5": Price(input=2.0, output=10.0),
    "claude-haiku-4-5": Price(input=1.0, output=5.0),
    "claude-fable-5-1": Price(input=10.0, output=50.0),
    "mlx-community/Qwen3-1.7B-4bit": Price(input=0.0007, output=0.0067),
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": Price(input=0.0014, output=0.015),
    "mlx-community/Qwen3-4B-Instruct-2507-6bit": Price(input=0.0014, output=0.021),
    "mlx-community/Qwen3-4B-Instruct-2507-8bit": Price(input=0.0014, output=0.027),
    "mlx-community/Qwen3-4B-Instruct-2507-bf16": Price(input=0.0014, output=0.050),
    "mlx-community/Qwen3-8B-4bit": Price(input=0.0028, output=0.025),
    # Sarvam lists ₹29.28 / ₹10.98 / ₹73.20 per million, converted at ₹95.5 and rounded up.
    "sarvam-105b": Price(input=0.31, output=0.77, cached_input=0.12),
}

# Billed per search, on top of the tokens its results add.
PRICE_PER_WEB_SEARCH = 0.01

# What the Message Batches API charges per token, as a share of the list price. Searches stay whole.
BATCH_DISCOUNT = 0.5


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    web_searches: int = 0,
    *,
    batched: bool = False,
    cache_write_1h_tokens: int = 0,
) -> float:
    """Dollar cost of one request. Unknown models cost nothing rather than crashing the script.
    `cache_write_1h_tokens` is the share of `cache_write_tokens` written at the 1-hour price."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    per_token = 1e-6 * (BATCH_DISCOUNT if batched else 1.0)
    return web_searches * PRICE_PER_WEB_SEARCH + per_token * (
        input_tokens * price.input
        + output_tokens * price.output
        + (cache_write_tokens - cache_write_1h_tokens) * price.cache_write
        + cache_write_1h_tokens * price.cache_write_1h
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
        batched=usage.batched,
        cache_write_1h_tokens=usage.cache_write_1h_tokens,
    )


def describe_usage(model: str, usage: Usage) -> str:
    """One line you can paste into LOG.md."""
    return (
        f"[{model}] in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_write={usage.cache_write_tokens} "
        + (f"cache_write_1h={usage.cache_write_1h_tokens} " if usage.cache_write_1h_tokens else "")
        + f"cache_read={usage.cache_read_tokens} "
        + (f"web_searches={usage.web_searches} " if usage.web_searches else "")
        + ("batched " if usage.batched else "")
        + f"cost=${usage_cost(model, usage):.5f}"
    )


def worst_case_usd(
    model: str,
    input_tokens: int,
    max_tokens: int,
    *,
    cached: bool = False,
    web_searches: int = 0,
    batched: bool = False,
    cache_ttl: CacheTtl = "5m",
) -> float:
    """The most one request can cost: the full output cap, and all input at the input price — or at
    the `cache_ttl` write price if `cached`, since a miss writes the whole prefix. Each web search
    adds its fee and `WEB_SEARCH_RESULT_TOKENS` of results: the one part estimated, not bounded."""
    input_tokens += web_searches * WEB_SEARCH_RESULT_TOKENS
    if cached:
        return cost_usd(
            model,
            0,
            max_tokens,
            cache_write_tokens=input_tokens,
            web_searches=web_searches,
            batched=batched,
            cache_write_1h_tokens=input_tokens if cache_ttl == "1h" else 0,
        )
    return cost_usd(model, input_tokens, max_tokens, web_searches=web_searches, batched=batched)


def affordable_output_tokens(
    model: str,
    input_tokens: int,
    limit_usd: float,
    *,
    cached: bool = False,
    web_searches: int = 0,
    cache_ttl: CacheTtl = "5m",
) -> int:
    """The largest `max_tokens` whose worst case, with this input, stays within `limit_usd`."""
    price = PRICES.get(model)
    if price is None:
        return 0
    spare = limit_usd - worst_case_usd(
        model, input_tokens, 0, cached=cached, web_searches=web_searches, cache_ttl=cache_ttl
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
    *,
    batched: bool = False,
    cache_ttl: CacheTtl = "5m",
) -> float:
    """Worst case for a whole run: cases x calls-per-case x `worst_case_usd`.

    `cached_tokens` prices a cached prefix as one write plus n-1 reads. Note the sign flips at n=1:
    caching a single row costs more than not caching it.
    """
    calls = n_cases * calls_per_case
    if not calls:
        return 0.0
    fresh = max(input_tokens_per_case - cached_tokens, 0)
    long = cached_tokens if cache_ttl == "1h" else 0
    first = cost_usd(
        model,
        fresh,
        max_tokens,
        cache_write_tokens=cached_tokens,
        batched=batched,
        cache_write_1h_tokens=long,
    )
    rest = cost_usd(model, fresh, max_tokens, cache_read_tokens=cached_tokens, batched=batched)
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
    batched: bool = False,
    cache_ttl: CacheTtl = "5m",
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
        model,
        input_tokens,
        max_tokens,
        cached=cached,
        web_searches=web_searches,
        batched=batched,
        cache_ttl=cache_ttl,
    )
    if worst > limit_usd:
        to = " written to the 1-hour cache" if cache_ttl == "1h" else " written to cache"
        written = to if cached else ""
        written += f", {web_searches} web searches" if web_searches else ""
        written += ", batched" if batched else ""
        raise BudgetExceeded(
            f"request could cost up to ${worst:.4f} ({input_tokens} input tokens{written} plus "
            f"max_tokens={max_tokens} on {model}), above the per-{scope} ceiling of "
            f"${limit_usd:.4f}. Trim the input, lower max_tokens, or raise "
            f"AIGENT_MAX_USD_PER_{scope.upper()} in .env."
        )
    return worst


@dataclass
class Budget:
    """Per-run accumulator. `charge` records an overrun in `tripped`; `add` also raises.

    Both bill before they trip, so `spent_usd` always includes the crossing call. `scope` picks
    which env var the message names. Safe to share between threads.
    """

    limit_usd: float = MAX_USD_PER_RUN
    spent_usd: float = 0.0
    scope: str = "run"
    tripped: str | None = None
    held_usd: float = 0.0  # worst cases of calls admitted and not yet billed
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def charge(self, model: str, usage: Usage) -> float:
        """Bill one call and return what it added. Never raises; sets `tripped` when over."""
        charged = usage_cost(model, usage)
        with self._lock:
            self.spent_usd += charged
            if self.spent_usd > self.limit_usd:
                self.tripped = (
                    f"{self.scope} has spent ${self.spent_usd:.4f}, above the per-{self.scope} "
                    f"ceiling of ${self.limit_usd:.4f}. Raise "
                    f"AIGENT_MAX_USD_PER_{self.scope.upper()} in .env if this was intended."
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
        with self._lock:
            if self.spent_usd + self.held_usd + worst_usd > self.limit_usd:
                held = (
                    f" with ${self.held_usd:.4f} held for calls in flight" if self.held_usd else ""
                )
                raise BudgetExceeded(
                    f"{self.scope} has spent ${self.spent_usd:.4f}{held}, and the next spending "
                    f"could cost up to ${worst_usd:.4f}, past the per-{self.scope} ceiling of "
                    f"${self.limit_usd:.4f}. Raise AIGENT_MAX_USD_PER_{self.scope.upper()} in "
                    ".env if this was intended."
                )
            if hold:
                self.held_usd += worst_usd

    def release(self, worst_usd: float) -> None:
        """Let go of a hold, once its call is billed or has failed."""
        with self._lock:
            self.held_usd = max(0.0, self.held_usd - worst_usd)
