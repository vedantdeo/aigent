"""The build's checks: the fundamentals found and equal, and a filing's own sums."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from aigent.facts.check import check_fundamentals, check_sums, tolerance
from aigent.facts.store import FilingRow, Writer, connect, sync_xbrl_facts, upsert_filing
from aigent.facts.xbrl import Fact

from .conftest import fundamental

WRITER = Writer("xbrl-loader@abc1234", "2026-10-22T12:00:00+05:30", "build-1")
YEAR = (date(2024, 4, 1), date(2025, 3, 31))


def _year(concept: str, value: float, decimals: int | None = -5) -> Fact:
    return Fact(concept, "duration", *YEAR, None, value, "INR", decimals)


def _load(
    conn: sqlite3.Connection,
    facts: list[Fact],
    period_end: date = date(2025, 3, 31),
    basis: str = "consolidated",
) -> int:
    fid = upsert_filing(
        conn,
        FilingRow("ITC", 2025, period_end, basis, "xbrl", f"{period_end}{basis}", "s", "t"),
    )
    sync_xbrl_facts(conn, fid, "ITC", basis, facts, WRITER)
    return fid


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return connect(tmp_path / "facts.sqlite")


@pytest.mark.parametrize(
    ("metric", "expected", "facts", "period_end", "basis", "status"),
    [
        pytest.param(
            "revenue",
            816127800000.0,
            [_year("RevenueFromOperations", 816127800000.0)],
            date(2025, 3, 31),
            "consolidated",
            "equal",
            id="found and equal",
        ),
        pytest.param(
            "revenue",
            816127800000.0,
            [_year("RevenueFromOperations", 816127840000.0)],
            date(2025, 3, 31),
            "consolidated",
            "equal",
            id="equal within the filed rounding",
        ),
        pytest.param(
            "profit",
            347466300000.0,
            [_year("ProfitOrLossAttributableToOwnersOfParent", 34746630000.0)],
            date(2025, 3, 31),
            "consolidated",
            "differs",
            id="a scale off by ten",
        ),
        pytest.param(
            "eps",
            27.79,
            [
                _year(
                    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations", 15.78, None
                )
            ],
            date(2025, 3, 31),
            "consolidated",
            "differs",
            id="a per-share figure differs",
        ),
        pytest.param(
            "revenue", 1.0, [], date(2025, 3, 31), "consolidated", "missing", id="not loaded"
        ),
        pytest.param(
            "revenue",
            1.0,
            [_year("RevenueFromOperations", 1.0)],
            date(2025, 3, 31),
            "standalone",
            "missing",
            id="only the standalone figure",
        ),
        pytest.param(
            "revenue",
            1.0,
            [_year("RevenueFromOperations", 1.0)],
            date(2025, 6, 30),
            "consolidated",
            "missing",
            id="only in a later quarter's filing",
        ),
        pytest.param(
            "dividend",
            14.35,
            [],
            date(2025, 3, 31),
            "consolidated",
            "not in xbrl",
            id="a metric XBRL does not file",
        ),
    ],
)
def test_check_fundamentals(
    conn: sqlite3.Connection,
    metric: str,
    expected: float,
    facts: list[Fact],
    period_end: date,
    basis: str,
    status: str,
) -> None:
    _load(conn, facts, period_end, basis)
    [checked] = check_fundamentals(conn, [fundamental("ITC", 2025, metric, expected)])
    assert checked.status == status, checked


def test_check_fundamentals_ignores_a_segments_figure(conn: sqlite3.Connection) -> None:
    _load(conn, [Fact("RevenueFromOperations", "duration", *YEAR, "FMCG", 1.0, "INR", -5)])
    [checked] = check_fundamentals(conn, [fundamental("ITC", 2025, "revenue", 1.0)])
    assert checked.status == "missing"


@pytest.mark.parametrize(
    ("values", "broken"),
    [
        pytest.param(
            {"Income": 110.0, "RevenueFromOperations": 100.0, "OtherIncome": 10.0},
            0,
            id="a sum that holds",
        ),
        pytest.param(
            {"Income": 110.0, "RevenueFromOperations": 100.0, "OtherIncome": 11.0},
            0,
            id="a sum that holds to the filed rounding",
        ),
        pytest.param(
            {"Income": 120.0, "RevenueFromOperations": 100.0, "OtherIncome": 10.0},
            1,
            id="a sum that breaks",
        ),
        pytest.param(
            {"Income": 120.0, "RevenueFromOperations": 100.0},
            0,
            id="a sum missing a term is not checked",
        ),
        pytest.param(
            {
                "Assets": 105.0,
                "CurrentAssets": 40.0,
                "NoncurrentAssets": 60.0,
                "NoncurrentAssetsClassifiedAsHeldForSale": 5.0,
            },
            0,
            id="assets held for sale are a term when filed",
        ),
        pytest.param(
            {
                "Assets": 105.0,
                "CurrentAssets": 40.0,
                "NoncurrentAssets": 60.0,
                "RegulatoryDeferralAccountDebitBalancesAndRelatedDeferredTaxAssets": 5.0,
            },
            0,
            id="regulatory deferral balances are a term when filed",
        ),
        pytest.param(
            {"Assets": 105.0, "CurrentAssets": 40.0, "NoncurrentAssets": 60.0},
            1,
            id="an optional term not filed counts as nothing",
        ),
    ],
)
def test_check_sums(conn: sqlite3.Connection, values: dict[str, float], broken: int) -> None:
    fid = _load(conn, [_year(c, v, 0) for c, v in values.items()])
    failures = check_sums(conn, fid)
    assert len(failures) == broken, failures


def test_check_sums_reads_each_period_and_balance_apart(conn: sqlite3.Connection) -> None:
    quarter = (date(2025, 1, 1), date(2025, 3, 31))
    fid = _load(
        conn,
        [
            _year("Income", 110.0, 0),
            _year("RevenueFromOperations", 100.0, 0),
            _year("OtherIncome", 10.0, 0),
            Fact("Income", "duration", *quarter, None, 30.0, "INR", 0),
            Fact("RevenueFromOperations", "duration", *quarter, None, 25.0, "INR", 0),
            Fact("OtherIncome", "duration", *quarter, None, 3.0, "INR", 0),
            Fact("Assets", "instant", None, date(2025, 3, 31), None, 9.0, "INR", 0),
            Fact("CurrentAssets", "instant", None, date(2025, 3, 31), None, 4.0, "INR", 0),
            Fact("NoncurrentAssets", "instant", None, date(2025, 3, 31), None, 5.0, "INR", 0),
            Fact("EquityAndLiabilities", "instant", None, date(2025, 3, 31), None, 9.0, "INR", 0),
            Fact("Income", "duration", *YEAR, "FMCG", 1.0, "INR", 0),
        ],
    )
    assert check_sums(conn, fid) == [
        "Income 30 != OtherIncome + RevenueFromOperations 28 (2025-01-01 to 2025-03-31)"
    ]


@pytest.mark.parametrize(
    ("decimals", "allowed"),
    [([-5], 50000.0), ([-5, -7], 5050000.0), ([None], 0.005), ([2, 2], 0.01)],
)
def test_tolerance_is_half_the_last_filed_digit_per_value(
    decimals: list[int | None], allowed: float
) -> None:
    assert tolerance(decimals) == pytest.approx(allowed)


@pytest.mark.parametrize(
    ("tier", "checker"),
    [
        pytest.param("xbrl-pdf", "pdf:results/ITC-2025-03-31.pdf", id="its document agrees"),
        pytest.param("xbrl", "vedant", id="a person checked it"),
    ],
)
def test_the_checks_read_a_checked_xbrl_fact(
    conn: sqlite3.Connection, tier: str, checker: str
) -> None:
    fid = _load(
        conn,
        [
            _year("RevenueFromOperations", 2.0, 0),
            _year("OtherIncome", 1.0, 0),
            _year("Income", 9.0, 0),
        ],
    )
    conn.execute("UPDATE facts SET tier = ?, verified_by = ?, verified_at = 't'", (tier, checker))
    [checked] = check_fundamentals(conn, [fundamental("ITC", 2025, "revenue", 2.0)])
    assert checked.status == "equal", checked
    assert len(check_sums(conn, fid)) == 1
