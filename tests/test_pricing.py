"""Prices as a table, then the two guards that read them.

The price list is pure arithmetic — one call in, one number out — so it belongs in a table rather
than in five functions that differ only by their inputs. The last two tests hold every structured
call in the package to invariant 1: its pre-flight counts the schema it sends.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import anthropic
import pytest
from anthropic.types import MessageTokensCount, Usage
from pydantic import BaseModel

import entropic
from entropic.config import DEFAULT_JUDGE_MODEL, DEFAULT_MODEL
from entropic.pricing import (
    PRICES,
    Budget,
    BudgetExceeded,
    assert_request_within_budget,
    check_request,
    cost_usd,
    estimate_eval_usd,
    worst_case_usd,
)

OPUS = "claude-opus-5"
PACKAGE = Path(entropic.__file__).parent


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


def test_both_default_models_are_priced() -> None:
    # cost_usd returns 0.0 for an unknown model rather than raising, which is right for a typo in
    # .env and very wrong for a typo in a default: every cost line in the repo would read $0.00000.
    for model in (DEFAULT_MODEL, DEFAULT_JUDGE_MODEL):
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


class _Counts:
    """Stands in for `client.messages`, keeping what the free count was asked to count."""

    def __init__(self) -> None:
        self.asked: dict[str, object] = {}

    def count_tokens(self, **kwargs: object) -> MessageTokensCount:
        self.asked = kwargs
        return MessageTokensCount(input_tokens=100)


class _Client:
    def __init__(self) -> None:
        self.messages = _Counts()


class _Schema(BaseModel):
    answer: str


def test_check_request_counts_the_schema_with_the_request() -> None:
    client = _Client()

    check_request(
        cast(anthropic.Anthropic, client),
        model=OPUS,
        max_tokens=64,
        messages=[{"role": "user", "content": "hi"}],
        output_format=_Schema,
    )

    assert client.messages.asked["output_format"] is _Schema


def _own_calls(function: ast.AST) -> list[ast.Call]:
    """The calls a function makes itself, not those of functions defined inside it."""
    calls: list[ast.Call] = []
    pending = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Call):
            calls.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return calls


def _schema_of(call: ast.Call) -> str | None:
    return next((ast.unparse(kw.value) for kw in call.keywords if kw.arg == "output_format"), None)


def _is_parse(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "parse"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "messages"
    )


def _parse_sites() -> list[tuple[str, str | None, list[str | None]]]:
    """Each function in the package calling `messages.parse`, its schema and its pre-flights'."""
    sites: list[tuple[str, str | None, list[str | None]]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            calls = _own_calls(node)
            preflights = [
                _schema_of(call)
                for call in calls
                if isinstance(call.func, ast.Name) and call.func.id == "check_request"
            ]
            for call in calls:
                if _is_parse(call):
                    where = f"{path.relative_to(PACKAGE)}:{node.name}"
                    sites.append((where, _schema_of(call), preflights))
    return sites


PARSE_SITES = _parse_sites()


def test_the_scan_finds_every_structured_call() -> None:
    """Five when this was written. Fewer means the scan broke, and the table below went vacuous."""
    assert len(PARSE_SITES) >= 5, [where for where, _, _ in PARSE_SITES]


@pytest.mark.parametrize(
    ("schema", "preflighted"),
    [pytest.param(schema, preflighted, id=where) for where, schema, preflighted in PARSE_SITES],
)
def test_a_structured_call_is_preflighted_with_the_schema_it_sends(
    schema: str | None, preflighted: list[str | None]
) -> None:
    """A schema is billed as input — `Extraction` alone is 1,148 tokens — so a pre-flight without
    it prices a cheaper request than the one sent."""
    assert schema is not None, "a parse call with no output_format"
    assert preflighted == [schema], f"sends {schema}, pre-flights with {preflighted}"
