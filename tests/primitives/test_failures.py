"""The error catalogue, checked for the two things a catalogue can get wrong.

It can lie about what costs money — a bad look in the one artifact whose subject is cost
discipline — and it can quietly stop provoking anything, passing while the API changes its mind
underneath. Neither the network nor a bill is needed to check either.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from anthropic import Anthropic
from anthropic.types import Usage

from entropic.config import MODEL
from entropic.pricing import Budget, BudgetExceeded
from entropic.primitives import failures as module
from entropic.primitives.failures import (
    FAILURES,
    Failure,
    _per_request_ceiling,
    _per_run_ceiling,
    _raises,
)

Provoke = Callable[[Anthropic], str]

# Constructing a client is offline and free; these two rows raise before they would ever use it.
UNUSED = Anthropic(api_key="unused")


def test_every_failure_is_named_once_and_says_what_it_costs() -> None:
    names = [failure.name for failure in FAILURES]
    assert len(set(names)) == len(names), "a duplicate name makes two rows indistinguishable"
    assert all(failure.when and failure.name for failure in FAILURES)


def test_a_provocation_that_stops_failing_is_reported_not_swallowed() -> None:
    """The rows exist to fail. One that starts succeeding is news — a deprecation reversed, a
    limit raised — so it has to read as a finding rather than pass quietly."""
    assert "no error" in _raises(lambda: None)
    assert _raises(lambda: 1 // 0).startswith("`ZeroDivisionError`")


@pytest.mark.parametrize(
    ("name", "provoke"),
    [("per request", _per_request_ceiling), ("per run", _per_run_ceiling)],
    ids=["per request", "per run"],
)
def test_our_own_ceilings_trip_without_touching_the_network(name: str, provoke: Provoke) -> None:
    """Two rows are pure arithmetic, which is why they are free and why they matter: the guard
    that costs nothing to check is the one you want sitting between you and the API."""
    detail = provoke(UNUSED)

    assert "BudgetExceeded" in detail
    assert "$" in detail, "a tripped guard names the number, not just the fact"


def test_the_per_run_ceiling_raises_rather_than_only_rendering() -> None:
    """`_raises` turns an exception into a table cell, so the cell could read right while the
    guard had stopped working. Check the guard itself, once."""
    with pytest.raises(BudgetExceeded, match=r"per-run ceiling"):
        Budget(limit_usd=0.01).add(MODEL, Usage(input_tokens=10**6, output_tokens=0))


def test_the_whole_catalogue_is_free_to_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing here may spend. Rows either provoke a rejected request — never billed, since no
    tokens reach the model — or raise before the network. A row that made a *successful* call
    would turn a document into a purchase, so `messages.parse` is the tell: nothing uses it."""
    source = Path(module.__file__ or "").read_text(encoding="utf-8")
    assert "messages.parse" not in source, "a successful structured call would be billed"
    assert "--yes" not in source, "no paid path means no gate to forget"

    ran: list[str] = []

    def watched(failure: Failure) -> Failure:
        def provoke(_: object) -> str:
            ran.append(failure.name)
            return "provoked"

        return Failure(failure.name, failure.when, provoke)

    monkeypatch.setattr(module, "FAILURES", tuple(watched(f) for f in FAILURES))
    monkeypatch.setattr(module, "get_client", lambda: UNUSED)
    module.main()

    assert ran == [failure.name for failure in FAILURES], "every row runs, in order"
    assert "| failure | provoked by |" in capsys.readouterr().out
