"""Prices as a table, then the two guards that read them.

The price list is pure arithmetic — one call in, one number out — so it belongs in a table rather
than in five functions that differ only by their inputs.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import pytest
from anthropic.types import Usage

from entropic.config import DEFAULT_JUDGE_MODEL, DEFAULT_MODEL, DEFAULT_SMALL_MODEL
from entropic.pricing import (
    PRICES,
    Budget,
    BudgetExceeded,
    assert_request_within_budget,
    cost_usd,
    estimate_eval_usd,
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
    ],
)
def test_the_price_list(charged: float, expected: float) -> None:
    assert charged == pytest.approx(expected)


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


def test_every_default_model_is_priced() -> None:
    # cost_usd returns 0.0 for an unknown model rather than raising, which is right for a typo in
    # .env and very wrong for a typo in a default: every cost line in the repo would read $0.00000.
    for model in (DEFAULT_MODEL, DEFAULT_JUDGE_MODEL, DEFAULT_SMALL_MODEL):
        assert model in PRICES, f"{model} has no price, so it would silently report as free"


def test_the_judge_does_not_default_to_the_model_it_grades() -> None:
    # Not a style preference: a model asked to grade its own output favours it.
    assert DEFAULT_JUDGE_MODEL != DEFAULT_MODEL
    assert PRICES[DEFAULT_JUDGE_MODEL].output < PRICES[DEFAULT_MODEL].output


def test_charge_records_the_trip_instead_of_raising() -> None:
    """What a batch wants: stop cleanly, keep the rows already paid for, say why.

    The scope is what picks the message, so this also pins that an eval is told about
    ENTROPIC_MAX_USD_PER_EVAL rather than about whichever knob happened to be first in the file.
    """
    budget = Budget(limit_usd=0.05, scope="eval")

    assert budget.charge(OPUS, _usage(1000, 1000)) == pytest.approx(0.03)
    assert budget.tripped is None

    assert budget.charge(OPUS, _usage(1000, 1000)) == pytest.approx(0.03)
    assert budget.spent_usd == pytest.approx(0.06)
    assert budget.tripped is not None
    assert "ENTROPIC_MAX_USD_PER_EVAL" in budget.tripped


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
            pytest.raises(BudgetExceeded, match="ENTROPIC_MAX_USD_PER_EVAL"),
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
