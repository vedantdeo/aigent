"""Prices as a table, then the two guards that read them.

The price list is pure arithmetic — one call in, one number out — so it belongs in a table rather
than in five functions that differ only by their inputs.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import pytest

from aigent.adapters import CLIENTS
from aigent.config import (
    WEB_SEARCH_RESULT_TOKENS,
)
from aigent.errors import BudgetExceeded
from aigent.messages import Usage
from aigent.pricing import (
    PRICES,
    Budget,
    affordable_output_tokens,
    assert_request_within_budget,
    cost_usd,
    estimate_eval_usd,
    usage_cost,
    worst_case_usd,
)

OPUS = "claude-opus-5"


@pytest.mark.parametrize(
    ("charged", "expected"),
    [
        pytest.param(cost_usd(OPUS, 10**6, 10**6), 30.0, id="1M input at $5 plus 1M output at $25"),
        pytest.param(
            cost_usd(OPUS, 0, 0, cache_write_tokens=10**6), 6.25, id="a cache write is 1.25x input"
        ),
        pytest.param(
            cost_usd(OPUS, 0, 0, cache_read_tokens=10**6), 0.5, id="a cache read is 0.1x input"
        ),
        pytest.param(
            cost_usd("claude-does-not-exist", 10, 10), 0.0, id="an unknown model is free, not fatal"
        ),
        pytest.param(
            worst_case_usd(OPUS, 1000, 4096),
            0.005 + 0.1024,
            id="the worst case charges the full output cap",
        ),
        pytest.param(
            worst_case_usd(OPUS, 1000, 4096, cached=True),
            0.00625 + 0.1024,
            id="a call that may write the cache writes all its input at worst",
        ),
        pytest.param(
            cost_usd(OPUS, 0, 0, web_searches=1000), 10.0, id="a thousand web searches are $10"
        ),
        pytest.param(
            usage_cost(OPUS, Usage(input_tokens=0, output_tokens=0, web_searches=3)),
            0.03,
            id="a response's web searches are billed with its tokens",
        ),
        pytest.param(
            worst_case_usd(OPUS, 1000, 4096, web_searches=2),
            0.005 + 0.1024 + 2 * (0.01 + WEB_SEARCH_RESULT_TOKENS * 5e-6),
            id="each search a call may run adds its fee and an allowance for its results",
        ),
    ],
)
def test_the_price_list(charged: float, expected: float) -> None:
    assert charged == pytest.approx(expected)


@pytest.mark.parametrize(
    ("input_tokens", "cached", "expected"),
    [
        pytest.param(
            10_000, False, 13_999, id="what $0.40 leaves after $0.05 of input, less a token"
        ),
        pytest.param(10_000, True, 13_499, id="less again when the input may be written to cache"),
        pytest.param(100_000, False, 0, id="none when the input alone is over the limit"),
    ],
)
def test_the_output_a_limit_affords(input_tokens: int, cached: bool, expected: int) -> None:
    affords = affordable_output_tokens(OPUS, input_tokens, 0.40, cached=cached)

    assert affords == expected
    worst = worst_case_usd(OPUS, input_tokens, affords, cached=cached)
    assert affords == 0 or worst <= 0.40, worst


def _usage(input_tokens: int, output_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def test_request_guard_passes_a_normal_week1_call() -> None:
    assert assert_request_within_budget(OPUS, 60, 1024, limit_usd=0.25) < 0.25


def test_request_guard_trips_on_runaway_input() -> None:
    with pytest.raises(BudgetExceeded, match="per-request ceiling"):
        assert_request_within_budget(OPUS, 40_000, 4096, limit_usd=0.25)


def test_request_guard_refuses_a_model_it_has_no_price_for() -> None:
    """A real model missing from `PRICES` counts without error and costs $0.00 by the arithmetic,
    so without this every ceiling would wave its calls through and bill them as free."""
    with pytest.raises(BudgetExceeded, match="has no price"):
        assert_request_within_budget("claude-opus-4-8", input_tokens=10, max_tokens=10)


def test_run_budget_accumulates_then_trips_after_the_crossing_call() -> None:
    budget = Budget(limit_usd=0.05)
    budget.add(OPUS, _usage(1000, 1000))  # $0.005 + $0.025 = $0.03
    assert budget.spent_usd == pytest.approx(0.03)
    with pytest.raises(BudgetExceeded, match="per-run ceiling"):
        budget.add(OPUS, _usage(1000, 1000))  # $0.06 > $0.05
    assert budget.spent_usd == pytest.approx(0.06)  # the crossing call was billed, so it is counted


def test_eval_estimate_is_the_per_request_worst_case_times_the_dataset() -> None:
    # 30 cases, one call each: 400 input at $5/M plus 1024 output at $25/M.
    one = worst_case_usd(OPUS, 400, 1024)
    assert estimate_eval_usd(OPUS, 30, 400, 1024) == pytest.approx(30 * one)
    # An LLM-as-judge grader doubles the calls, and the estimate has to know that.
    assert estimate_eval_usd(OPUS, 30, 400, 1024, calls_per_case=2) == pytest.approx(60 * one)


def test_a_cached_prefix_is_written_once_and_read_thereafter() -> None:
    """The estimate has to model the cache, or it refuses a cached run for costing what the
    uncached one costs — blocking the experiment on the arithmetic it exists to check."""
    plain = estimate_eval_usd(OPUS, 50, 1700, 384)
    cached = estimate_eval_usd(OPUS, 50, 1700, 384, cached_tokens=1600)

    assert cached < plain, "50 rows behind one breakpoint is cheaper than 50 fresh ones"
    # One write at a 25% premium, 49 reads at a tenth, and the uncached tail every time.
    first = cost_usd(OPUS, 100, 384, cache_write_tokens=1600)
    rest = cost_usd(OPUS, 100, 384, cache_read_tokens=1600)
    assert cached == pytest.approx(first + 49 * rest)


def test_caching_one_row_costs_more_than_not_caching_it() -> None:
    """A breakpoint is a bet that the prefix gets reused: you pay 25% extra to write it. A smoke
    run of one row is the one place it cannot pay off, so it is the wrong place to test caching."""
    assert estimate_eval_usd(OPUS, 1, 1700, 384, cached_tokens=1600) > estimate_eval_usd(
        OPUS, 1, 1700, 384
    )
    # Two rows is already enough: one read saves more than the write premium cost.
    assert estimate_eval_usd(OPUS, 2, 1700, 384, cached_tokens=1600) < estimate_eval_usd(
        OPUS, 2, 1700, 384
    )


def test_an_estimate_with_no_cached_prefix_is_the_arithmetic_it_always_was() -> None:
    """`cached_tokens=0` is the default and must not shift a single existing number."""
    assert estimate_eval_usd(OPUS, 30, 400, 1024, cached_tokens=0) == pytest.approx(
        30 * worst_case_usd(OPUS, 400, 1024)
    )
    assert estimate_eval_usd(OPUS, 0, 400, 1024, cached_tokens=400) == 0.0


def test_every_model_any_client_names_is_priced() -> None:
    # cost_usd returns 0.0 for an unknown model rather than raising, which is right for a typo in
    # .env and very wrong for a typo in a client's config: every cost line would read $0.00000.
    # A local model is priced too, at what its hour of machine costs — free is a price, not a gap.
    for client in CLIENTS.values():
        for model in (client.model, client.judge_model, client.small_model):
            assert model in PRICES, f"{client.name} names {model}, which would report as free"


def test_no_client_lets_the_judge_grade_the_model_it_is() -> None:
    """Not a style preference: a model asked to grade its own output favours it.

    This used to also assert the judge was the cheaper of the two, which held while every client
    was hosted and the judge was the smaller model. It is false locally and rightly so: the local
    judge is a 7B grading a 4B, and costs more per token *because* it is bigger and slower. Price
    was never the rule; being a different model is.
    """
    for client in CLIENTS.values():
        assert client.judge_model != client.model, client.name


def test_charge_records_the_trip_instead_of_raising() -> None:
    """What a batch wants: stop cleanly, keep the rows already paid for, say why.

    The scope is what picks the message, so this also pins that an eval is told about
    AIGENT_MAX_USD_PER_EVAL rather than about whichever knob happened to be first in the file.
    """
    budget = Budget(limit_usd=0.05, scope="eval")

    assert budget.charge(OPUS, _usage(1000, 1000)) == pytest.approx(0.03)
    assert budget.tripped is None

    assert budget.charge(OPUS, _usage(1000, 1000)) == pytest.approx(0.03)
    assert budget.spent_usd == pytest.approx(0.06)
    assert budget.tripped is not None
    assert "AIGENT_MAX_USD_PER_EVAL" in budget.tripped


def test_add_is_charge_plus_a_raise_so_the_two_cannot_drift() -> None:
    charged = Budget(limit_usd=0.01)
    raised = Budget(limit_usd=0.01)

    charged.charge(OPUS, _usage(1000, 1000))
    with pytest.raises(BudgetExceeded) as caught:
        raised.add(OPUS, _usage(1000, 1000))

    assert str(caught.value) == charged.tripped
    assert raised.spent_usd == charged.spent_usd


@pytest.mark.parametrize(
    ("spent", "worst", "expectation"),
    [
        pytest.param(0.0, 0.04, nullcontext(), id="nothing spent and the call fits"),
        pytest.param(0.03, 0.01, nullcontext(), id="spent plus worst case lands under"),
        pytest.param(
            0.03,
            0.03,
            pytest.raises(BudgetExceeded, match="AIGENT_MAX_USD_PER_EVAL"),
            id="spent plus worst case lands over",
        ),
        pytest.param(0.0, 0.06, pytest.raises(BudgetExceeded), id="the call alone could cross it"),
    ],
)
def test_admit_refuses_what_could_cross_the_ceiling_before_it_is_spent(
    spent: float, worst: float, expectation: AbstractContextManager[object]
) -> None:
    budget = Budget(limit_usd=0.05, spent_usd=spent, scope="eval")

    with expectation:
        budget.admit(worst)
    assert budget.spent_usd == spent, "admitting checks; nothing is billed until a call returns"


def test_a_held_call_counts_until_it_is_released() -> None:
    """Calls in flight have not been billed, so admission holds their worst case: two that each
    fit alone are not both admitted while the first is still running."""
    budget = Budget(limit_usd=0.05, scope="run")

    budget.admit(0.03, hold=True)
    with pytest.raises(BudgetExceeded, match="held for calls in flight"):
        budget.admit(0.03)
    budget.release(0.03)
    budget.admit(0.03)

    assert (budget.held_usd, budget.spent_usd) == (0.0, 0.0), "checking holds and bills nothing"
