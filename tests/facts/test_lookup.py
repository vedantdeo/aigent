"""Looking facts up: periods as dates, `what` matched by label, the exact fetch, and the log."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from aigent.facts.lookup import (
    ConceptIndex,
    Lookup,
    NotFound,
    Period,
    concept_index,
    concept_texts,
    find,
    segment_of,
    span,
)
from aigent.facts.store import connect

from ..conftest import DECEMBER_URL, MARCH_URL, BagOfWordsEmbedder


@pytest.mark.parametrize(
    ("period", "start", "end"),
    [
        ("year", date(2024, 4, 1), date(2025, 3, 31)),
        ("Q1", date(2024, 4, 1), date(2024, 6, 30)),
        ("Q2", date(2024, 7, 1), date(2024, 9, 30)),
        ("Q3", date(2024, 10, 1), date(2024, 12, 31)),
        ("Q4", date(2025, 1, 1), date(2025, 3, 31)),
        ("H1", date(2024, 4, 1), date(2024, 9, 30)),
        ("9M", date(2024, 4, 1), date(2024, 12, 31)),
    ],
)
def test_a_period_of_fy25_is_its_dates(period: Period, start: date, end: date) -> None:
    assert span(2025, period) == (start, end)


@pytest.fixture
def conn(facts_db: Path) -> sqlite3.Connection:
    return connect(facts_db)


def test_a_concept_is_matched_by_its_label_and_its_aliases(conn: sqlite3.Connection) -> None:
    texts = concept_texts(conn)
    assert texts["RevenueFromOperations"] == "Revenue from operations; Sales"
    index = ConceptIndex(texts, BagOfWordsEmbedder())
    assert index.match("sales")[0] == "RevenueFromOperations"
    assert index.match("total assets")[0] == "Assets"


@pytest.mark.parametrize(
    ("period", "basis", "value", "start"),
    [
        pytest.param("year", "consolidated", 816.0, "2024-04-01", id="a year"),
        pytest.param("Q4", "consolidated", 204.0, "2025-01-01", id="a quarter alone"),
        pytest.param("Q3", "consolidated", 203.5, "2024-10-01", id="another filing's quarter"),
        pytest.param("9M", "consolidated", 612.0, "2024-04-01", id="nine months"),
        pytest.param("year", "standalone", 700.0, "2024-04-01", id="the other basis"),
    ],
)
def test_a_fact_is_fetched_for_exactly_its_period_and_basis(
    conn: sqlite3.Connection, period: Period, basis: str, value: float, start: str
) -> None:
    [found] = find(conn, "ITC", 2025, period, basis, None, ["RevenueFromOperations"], "r")
    assert (found.value, found.period_start, found.basis) == (value, start, basis)


@pytest.mark.parametrize(
    ("ticker", "period", "basis"),
    [
        pytest.param("WIPRO", "year", "consolidated", id="another company"),
        pytest.param("ITC", "H1", "consolidated", id="a period not filed"),
        pytest.param("ITC", "Q4", "standalone", id="a basis not filed for that period"),
    ],
)
def test_nothing_is_fetched_for_another_company_period_or_basis(
    conn: sqlite3.Connection, ticker: str, period: Period, basis: str
) -> None:
    assert find(conn, ticker, 2025, period, basis, None, ["RevenueFromOperations"], "r") == []


@pytest.mark.parametrize(
    ("period", "value", "end"),
    [
        pytest.param("year", 900.0, "2025-03-31", id="at the year's end"),
        pytest.param("9M", 880.0, "2024-12-31", id="at the nine months' end"),
    ],
)
def test_a_balance_is_as_at_the_period_s_end(
    conn: sqlite3.Connection, period: Period, value: float, end: str
) -> None:
    [found] = find(conn, "ITC", 2025, period, "consolidated", None, ["Assets"], "r")
    assert (found.value, found.period_start, found.period_end) == (value, None, end)
    assert found.as_json()["period"] == {"as_at": end}


def test_a_segment_fact_needs_its_segment(conn: sqlite3.Connection) -> None:
    concepts = ["SegmentRevenue"]
    assert find(conn, "ITC", 2025, "year", "consolidated", None, concepts, "r") == []
    [found] = find(conn, "ITC", 2025, "year", "consolidated", "Agri Business", concepts, "r")
    assert (found.value, found.cite_as) == (
        201.0,
        "FACT:ITC:2025:consolidated:year:SegmentRevenue:Agri Business",
    )


def test_a_fact_carries_what_to_cite_and_where_it_was_filed(conn: sqlite3.Connection) -> None:
    [found] = find(conn, "ITC", 2025, "Q3", "consolidated", None, ["RevenueFromOperations"], "r")
    assert found.as_json() == {
        "cite_as": "FACT:ITC:2025:consolidated:Q3:RevenueFromOperations",
        "label": "Revenue from operations",
        "value": 203.5,
        "unit": "INR",
        "period": {"from": "2024-10-01", "to": "2024-12-31"},
        "basis": "consolidated",
        "segment": None,
        "tier": "xbrl-ai",
        "source": f"NSE XBRL results filing, period ended 2024-12-31: {DECEMBER_URL}",
    }


def test_a_balance_filed_twice_is_answered_from_the_earliest_filing(
    conn: sqlite3.Connection,
) -> None:
    concepts = ["CashAndCashEquivalentsCashFlowStatement"]
    # The opening balance at 2024-03-31 is a Q4-of-FY24 instant; FY24's year ends that day.
    [found] = find(conn, "ITC", 2024, "year", "consolidated", None, concepts, "r")
    assert (found.value, found.source.endswith(DECEMBER_URL)) == (5.0, True)


@pytest.mark.parametrize("run_id", ["other-run", "this-run"])
def test_an_unchecked_read_is_found_in_any_run_for_the_tool_to_hold_back(
    conn: sqlite3.Connection, run_id: str
) -> None:
    [found] = find(conn, "ITC", 2025, "year", "consolidated", None, ["DividendPerShare"], run_id)
    assert found.tier == "pdf-read-ai"


@pytest.mark.parametrize(
    ("run_id", "value"),
    [
        pytest.param("this-run", 2.0, id="a checked fact before this run's own read"),
        pytest.param("mine", 2.0, id="a checked fact before another run's"),
    ],
)
def test_a_checked_fact_comes_first(conn: sqlite3.Connection, run_id: str, value: float) -> None:
    with conn:
        for tier, by, val, run in (
            ("pdf-read-ai", None, 1.0, run_id),
            ("pdf-read", "vedant", 2.0, "old"),
            ("pdf-read-ai", None, 3.0, "this-run"),
        ):
            conn.execute(
                """INSERT INTO facts (ticker, basis, concept, label, period_kind, period_start,
                          period_end, value, unit, tier, written_by, written_at, run_id,
                          verified_by, verified_at)
                   VALUES ('ITC', 'consolidated', 'Extra', 'Extra', 'duration', '2024-04-01',
                           '2025-03-31', ?, 'INR', ?, 'w', 't', ?, ?, ?)""",
                (val, tier, run, by, None if by is None else "t"),
            )
    [found] = find(conn, "ITC", 2025, "year", "consolidated", None, ["Extra"], run_id)
    assert found.value == value, found


def test_this_runs_own_unchecked_read_comes_before_another_runs(conn: sqlite3.Connection) -> None:
    with conn:
        for val, run in ((1.0, "other"), (2.0, "this-run")):
            conn.execute(
                """INSERT INTO facts (ticker, basis, concept, label, period_kind, period_start,
                          period_end, value, unit, tier, written_by, written_at, run_id)
                   VALUES ('ITC', 'consolidated', 'Extra', 'Extra', 'duration', '2024-04-01',
                           '2025-03-31', ?, 'INR', 'pdf-read-ai', 'w', 't', ?)""",
                (val, run),
            )
    [found] = find(conn, "ITC", 2025, "year", "consolidated", None, ["Extra"], "this-run")
    assert found.value == 2.0, found


def test_at_most_k_facts_one_per_concept_in_the_order_matched(conn: sqlite3.Connection) -> None:
    concepts = [
        "Assets",
        "Missing",
        "RevenueFromOperations",
        "ProfitOrLossAttributableToOwnersOfParent",
    ]
    found = find(conn, "ITC", 2025, "year", "consolidated", None, concepts, "r", k=2)
    assert [f.concept for f in found] == ["Assets", "RevenueFromOperations"]


@pytest.mark.parametrize(
    ("asked", "named"),
    [
        pytest.param(None, None, id="no segment asked"),
        pytest.param("  ", None, id="a blank one"),
        pytest.param("agri business", "Agri Business", id="the same words in another case"),
        pytest.param("Agri", "Agri Business", id="words one segment holds"),
        pytest.param("others", "Others", id="a name another segment's holds"),
    ],
)
def test_a_segment_is_named_as_the_company_files_it(
    conn: sqlite3.Connection, asked: str | None, named: str | None
) -> None:
    assert segment_of(conn, "ITC", "consolidated", date(2025, 3, 31), asked) == named


@pytest.mark.parametrize(
    "asked", [pytest.param("FMCG", id="words two segments hold"), pytest.param("Hotels", id="none")]
)
def test_a_segment_not_named_once_is_refused_with_what_is_filed(
    conn: sqlite3.Connection, asked: str
) -> None:
    with pytest.raises(NotFound, match="Agri Business; FMCG - Cigarettes; FMCG - Others; Others"):
        segment_of(conn, "ITC", "consolidated", date(2025, 3, 31), asked)


def _logged(db: Path) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT run_id, asked_by, asked_at, ticker, basis, period, what, concept, found "
            "FROM lookups ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def test_a_lookup_answers_and_is_logged(facts_lookup: Lookup, facts_db: Path) -> None:
    found = facts_lookup("ITC", 2025, "revenue from operations", period="Q4")
    assert found[0].value == 204.0
    assert _logged(facts_db) == [
        (
            "this-run",
            "test",
            "2026-10-23T10:00:00",
            "ITC",
            "consolidated",
            "2025 Q4",
            "revenue from operations",
            "RevenueFromOperations",
            1,
        )
    ]


def test_a_lookup_with_nothing_to_answer_names_the_closest_labels_and_is_logged(
    facts_lookup: Lookup, facts_db: Path
) -> None:
    with pytest.raises(NotFound, match="Closest labels filed: Revenue from operations"):
        facts_lookup("WIPRO", 2025, "revenue from operations")
    [logged] = _logged(facts_db)
    assert (logged[3], logged[7], logged[8]) == ("WIPRO", "RevenueFromOperations", 0)


def test_a_refused_segment_is_logged_too(facts_lookup: Lookup, facts_db: Path) -> None:
    with pytest.raises(NotFound, match="no single segment"):
        facts_lookup("ITC", 2025, "segment revenue", segment="Hotels")
    assert [row[8] for row in _logged(facts_db)] == [0]


def test_a_database_not_built_is_said_so(tmp_path: Path) -> None:
    lookup = Lookup(tmp_path / "none.sqlite", concept_index(BagOfWordsEmbedder()), "r", "t")
    with pytest.raises(NotFound, match="not built"):
        lookup("ITC", 2025, "revenue")
    assert not (tmp_path / "none.sqlite").exists()


def test_the_index_is_built_once(facts_db: Path) -> None:
    built: list[int] = []
    make = concept_index(BagOfWordsEmbedder())

    def counting(conn: sqlite3.Connection) -> ConceptIndex:
        built.append(1)
        return make(conn)

    lookup = Lookup(facts_db, counting, "r", "t")
    lookup("ITC", 2025, "assets")
    lookup("ITC", 2025, "sales")
    assert built == [1]


def test_the_march_filing_is_named_as_the_source_of_a_year(facts_lookup: Lookup) -> None:
    [first, *_] = facts_lookup("ITC", 2025, "revenue from operations")
    assert first.source.endswith(MARCH_URL)
