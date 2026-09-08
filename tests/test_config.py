from __future__ import annotations

import pytest
from anthropic.types import Usage

from entropic.config import (
    Budget,
    BudgetExceeded,
    assert_request_within_budget,
    cost_usd,
    worst_case_usd,
)


def test_opus_pricing() -> None:
    # 1M input at $5 and 1M output at $25
    assert cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_cache_multipliers() -> None:
    write = cost_usd("claude-opus-5", 0, 0, cache_write_tokens=1_000_000)
    read = cost_usd("claude-opus-5", 0, 0, cache_read_tokens=1_000_000)
    assert write == pytest.approx(6.25)
    assert read == pytest.approx(0.5)


def test_unknown_model_is_free_not_fatal() -> None:
    assert cost_usd("claude-does-not-exist", 10, 10) == 0.0


def test_worst_case_charges_the_full_output_cap() -> None:
    # 1000 input at $5/M plus 4096 output at $25/M
    assert worst_case_usd("claude-opus-5", 1000, 4096) == pytest.approx(0.005 + 0.1024)


def test_request_guard_passes_a_normal_week1_call() -> None:
    assert assert_request_within_budget("claude-opus-5", 60, 1024, limit_usd=0.25) < 0.25


def test_request_guard_trips_on_runaway_input() -> None:
    with pytest.raises(BudgetExceeded, match="per-request ceiling"):
        assert_request_within_budget("claude-opus-5", 40_000, 4096, limit_usd=0.25)


def _usage(input_tokens: int, output_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def test_run_budget_accumulates_then_trips_after_the_crossing_call() -> None:
    budget = Budget(limit_usd=0.05)
    budget.add("claude-opus-5", _usage(1000, 1000))  # $0.005 + $0.025 = $0.03
    assert budget.spent_usd == pytest.approx(0.03)
    with pytest.raises(BudgetExceeded, match="per-run ceiling"):
        budget.add("claude-opus-5", _usage(1000, 1000))  # $0.06 > $0.05
    assert budget.spent_usd == pytest.approx(0.06)  # the crossing call was billed, so it is counted
