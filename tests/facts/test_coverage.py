"""The coverage eval: the three sources loaded, figures held by concept or value, found by rank,
exact, summarised and reported; its lookups never kept."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import cast

import pytest

from aigent.facts import __main__ as cli
from aigent.facts.cfg_facts import COVERAGE_FOUND_AT
from aigent.facts.coverage import (
    TABLE_FIGURES,
    TABLE_QUESTIONS,
    Figure,
    Left,
    Phrasing,
    Scored,
    evaluate,
    fundamentals_figures,
    holdings,
    score,
    summarise,
    table_figures,
    teacher_figures,
    to_markdown,
    tolerance_of,
    write_report,
)
from aigent.facts.lookup import Basis, Found, Lookup, NotFound, Period, concept_index
from aigent.facts.store import FilingRow, apply_labels, connect, sync_xbrl_facts, upsert_filing
from aigent.facts.taxonomy import Labels
from aigent.facts.xbrl import Fact

from ..conftest import FACTS_NOW, FACTS_WRITER, BagOfWordsEmbedder
from .conftest import fundamental

_FY25 = (date(2024, 4, 1), date(2025, 3, 31))
_END = date(2025, 3, 31)


def _d(
    concept: str,
    value: float,
    unit: str = "INR",
    decimals: int | None = -5,
    segment: str | None = None,
) -> Fact:
    return Fact(concept, "duration", _FY25[0], _FY25[1], segment, value, unit, decimals)


CONSOLIDATED = [
    _d("RevenueFromOperations", 816_127_800_000),
    _d("ProfitOrLossAttributableToOwnersOfParent", 347_000_000_000),
    _d(
        "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        16.42,
        "INRPerShare",
        None,
    ),
    Fact("Assets", "instant", None, _END, None, 880_906_800_000, "INR", -5),
    _d("SegmentRevenue", 358_935_700_000, segment="FMCG - Cigarettes"),
    _d("PurchaseOfPropertyPlantAndEquipment", 37_068_000_000),
    _d("CurrentRatio", 3.03, "pure", None),
    _d("OtherAdjustments", 30_000_000, decimals=-7),
]
LABELS = {
    "Assets": Labels("Total assets"),
    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations": Labels(
        "Basic earnings per share"
    ),
    "CurrentRatio": Labels("Current ratio"),
    "OtherAdjustments": Labels("Other adjustments"),
    "ProfitOrLossAttributableToOwnersOfParent": Labels("Profit attributable to owners of parent"),
    "PurchaseOfPropertyPlantAndEquipment": Labels("Purchase of property plant and equipment"),
    "RevenueFromOperations": Labels("Revenue from operations"),
    "SegmentRevenue": Labels("Segment revenue"),
}


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "facts.sqlite"
    conn = connect(path)
    with conn:
        for basis, facts in (
            ("consolidated", CONSOLIDATED),
            ("standalone", [_d("RevenueFromOperations", 700_000_000_000)]),
        ):
            row = FilingRow(
                "ITC", 2025, _END, basis, "xbrl", f"https://nse.example/{basis}", "s", "t"
            )
            sync_xbrl_facts(conn, upsert_filing(conn, row), "ITC", basis, facts, FACTS_WRITER)
        conn.execute(
            """INSERT INTO facts (ticker, basis, concept, label, period_kind, period_start,
                      period_end, value, unit, tier, written_by, written_at, run_id)
               VALUES ('ITC', 'consolidated', 'DividendPerShare', 'Dividend per share',
                       'duration', '2024-04-01', '2025-03-31', 14.35, 'INRPerShare',
                       'pdf-read-ai', 'model|agent@x', '2026-10-23', 'other-run')"""
        )
        apply_labels(conn, LABELS)
    conn.close()
    return path


@pytest.fixture
def conn(db: Path) -> Iterator[sqlite3.Connection]:
    c = connect(db)
    yield c
    c.close()


@pytest.fixture
def lookup(db: Path) -> Lookup:
    return Lookup(
        db, concept_index(BagOfWordsEmbedder()), "coverage-test", "test", lambda: FACTS_NOW
    )


def by_value(
    value: float, units: tuple[str, ...] = ("INR",), tol: float = 0.0, **asks: str
) -> Figure:
    words = cast(
        dict[Phrasing, str], {"asked": "revenue from operations", "word": "revenue", **asks}
    )
    return Figure("f", "tables", "ITC", 2025, value, tol, words, "value", units=units)


def by_concept(concept: str | None, value: float) -> Figure:
    words = cast(dict[Phrasing, str], {"asked": "revenue from operations", "word": "revenue"})
    return Figure(
        "ITC:2025:revenue", "fundamentals", "ITC", 2025, value, 0.0, words, "concept", concept
    )


# The sources.


def test_a_fundamental_is_matched_by_its_concept_and_asked_by_its_definition() -> None:
    revenue, dividend = fundamentals_figures(
        [fundamental("ITC", 2025, "revenue", 8.0), fundamental("ITC", 2025, "dividend", 14.35)]
    )
    assert (revenue.id, revenue.match, revenue.concept) == (
        "ITC:2025:revenue",
        "concept",
        "RevenueFromOperations",
    )
    assert revenue.asks == {"asked": "revenue from operations", "word": "revenue"}
    assert (dividend.match, dividend.concept) == ("concept", None)


def _tables(
    tmp_path: Path, entries: dict[str, dict[str, object]]
) -> tuple[list[Figure], list[Left]]:
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        "".join(
            json.dumps({"id": i, "input": {"question": f"what is {i}?"}}) + "\n"
            for i in ("a", "b", "c", "d")
        )
    )
    figures = tmp_path / "f.json"
    figures.write_text(json.dumps({"figures": entries}))
    return table_figures(questions, figures)


def _entry(ticker: str = "ITC", year: int = 2025, excluded: str | None = None) -> dict[str, object]:
    return {
        "ticker": ticker,
        "year": year,
        "metric": "total assets",
        "printed": "88090.68",
        "scale": 1e7,
        "unit": "INR",
        "excluded": excluded,
    }


def test_a_table_figure_is_its_printed_number_scaled_and_asked_as_written(tmp_path: Path) -> None:
    [figure], left = _tables(tmp_path, {"a": _entry()})
    assert left == []
    assert figure.value == pytest.approx(880_906_800_000)
    assert figure.tolerance == pytest.approx(50_000)
    assert (figure.match, figure.units) == ("value", ("INR",))
    assert figure.asks == {"asked": "what is a?", "word": "total assets"}


def test_a_table_figure_left_out_says_why(tmp_path: Path) -> None:
    scored, left = _tables(
        tmp_path,
        {"a": _entry(excluded="a series"), "b": _entry(ticker="TCS"), "c": _entry(year=2016)},
    )
    assert scored == []
    assert [(x.id, x.reason) for x in left] == [
        ("a", "a series"),
        ("b", "not a company the database holds"),
        ("c", "not a year the database holds"),
    ]


def test_a_table_figure_for_no_question_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="z is not a table question"):
        _tables(tmp_path, {"z": _entry()})


def _digits(text: str) -> str:
    return re.sub(r"[,()\s]", "", text)


def test_every_table_figure_is_printed_in_its_question_s_quote_for_its_company() -> None:
    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in TABLE_QUESTIONS.read_text().splitlines()
        if line.strip()
    }
    entries = cast(dict[str, dict[str, object]], json.loads(TABLE_FIGURES.read_text())["figures"])
    fields = ["ticker", "year", "metric", "printed", "scale", "unit", "excluded"]
    for id_, entry in entries.items():
        assert list(entry) == fields, id_
        assert rows[id_]["tags"][0].startswith(f"{entry['ticker']}-"), id_
        if entry["excluded"] is None:
            assert str(entry["printed"]).lstrip("-") in _digits(rows[id_]["expected"]["quote"]), id_
    assert set(entries) == set(rows)


@pytest.mark.parametrize(
    ("call", "value", "expected"),
    [
        pytest.param("834.05 * 10**7", 8.3405e9, 5e4, id="crore with two decimals"),
        pytest.param("1,234 * 10**7", 1.234e10, 5e6, id="grouping commas"),
        pytest.param("-33 * 10**7", -3.3e8, 5e6, id="a negative amount"),
        pytest.param("1648224221 * 1000", 1.648224221e12, 500.0, id="thousands"),
        pytest.param("5.5 * 10**5 * 10**2", 5.5e7, 5e5, id="two multipliers"),
        pytest.param("(1.5 + 2.25) * 10**7", 3.75e7, 5e4, id="a sum, by its finest term"),
        pytest.param("1710425700000", 1.7104257e12, 0.5, id="a whole number of rupees"),
        pytest.param(None, 16.42, 0.005, id="no call: from the value"),
        pytest.param("23 / 100", 0.23, None, id="a rate"),
        pytest.param("x * 10**7", 1.0, None, id="a name"),
        pytest.param("5 * 2.5", 12.5, None, id="a multiplier that is not whole"),
        pytest.param("5 *", 5.0, None, id="not an expression"),
        pytest.param("1e3 * 10**7", 1e10, None, id="exponent notation"),
    ],
)
def test_a_teacher_label_s_precision_is_its_last_written_digit(
    call: str | None, value: float, expected: float | None
) -> None:
    got = tolerance_of(call, value)
    assert got == (None if expected is None else pytest.approx(expected)), got


def _teacher(
    tmp_path: Path, figures: list[dict[str, object]], question: str = "q?"
) -> tuple[list[Figure], list[Left]]:
    path = tmp_path / "t.jsonl"
    row = {"chunk_id": "BAJAJ-AUTO-FY24#0004", "question": question, "figures": figures}
    path.write_text(json.dumps(row) + "\n\n")
    return teacher_figures(path)


def _label(metric: str | None = "revenue", **change: object) -> dict[str, object]:
    return {
        "metric": metric,
        "value": 4.4685e11,
        "unit": "INR",
        "year": 2024,
        "description": None,
        "call": "44685 * 10**7",
        **change,
    }


@pytest.mark.parametrize(
    ("label", "source", "reason"),
    [
        pytest.param(_label("other"), "teacher", "no named metric", id="no named metric"),
        pytest.param(_label(None), "teacher", "no named metric", id="no metric at all"),
        pytest.param(_label(year=None), "teacher-results", "no year or no value", id="no year"),
        pytest.param(_label(value=None), "teacher-results", "no year or no value", id="no value"),
        pytest.param(
            _label(year=2023), "teacher-results", "not a year the database holds", id="FY23"
        ),
        pytest.param(
            _label(unit="USD"), "teacher-results", "not an amount in rupees", id="dollars"
        ),
        pytest.param(
            _label(description="change"),
            "teacher-results",
            "not an amount in rupees",
            id="a change",
        ),
        pytest.param(
            _label(value=0.0), "teacher-results", "zero, which any nil fact would match", id="zero"
        ),
        pytest.param(
            _label(call="23 / 100"),
            "teacher-results",
            "its precision cannot be read from its call",
            id="a rate's call",
        ),
        pytest.param(_label("volume"), "teacher-other", None, id="a metric results do not file"),
        pytest.param(_label("cash_flow"), "teacher-results", None, id="a results metric"),
    ],
)
def test_a_teacher_label_is_scored_or_left_out_with_its_reason(
    tmp_path: Path, label: dict[str, object], source: str, reason: str | None
) -> None:
    scored, left = _teacher(tmp_path, [label])
    if reason is None:
        [figure] = scored
        assert (figure.source, figure.ticker, figure.year, figure.id) == (
            source,
            "BAJAJ-AUTO",
            2024,
            "BAJAJ-AUTO-FY24#0004/0",
        )
        assert figure.asks["word"] == str(label["metric"]).replace("_", " ")
    else:
        assert scored == []
        assert [(x.source, x.reason) for x in left] == [(source, reason)]


@pytest.mark.parametrize(
    ("metric", "units"),
    [("eps", ("INRPerShare",)), ("dividend", ("INR", "INRPerShare")), ("revenue", ("INR",))],
)
def test_a_teacher_label_is_matched_in_its_metric_s_units(
    tmp_path: Path, metric: str, units: tuple[str, ...]
) -> None:
    [figure], _ = _teacher(tmp_path, [_label(metric)])
    assert figure.units == units


def test_a_teacher_figure_repeated_with_its_question_counts_once(tmp_path: Path) -> None:
    scored, left = _teacher(
        tmp_path, [_label(), _label(), _label(value=4.5e11, call="45000 * 10**7")]
    )
    assert [f.id for f in scored] == ["BAJAJ-AUTO-FY24#0004/0", "BAJAJ-AUTO-FY24#0004/2"]
    assert [x.reason for x in left] == ["repeats a figure already counted"]


# Held.


@pytest.mark.parametrize(
    ("figure", "held"),
    [
        pytest.param(
            by_concept("RevenueFromOperations", 1.0),
            [("RevenueFromOperations", "consolidated", None)],
            id="by concept, consolidated only",
        ),
        pytest.param(by_concept(None, 1.0), [], id="a concept not filed"),
        pytest.param(by_concept("Goodwill", 1.0), [], id="a concept this company does not file"),
        pytest.param(
            by_value(816_127_800_000),
            [("RevenueFromOperations", "consolidated", None)],
            id="by value",
        ),
        pytest.param(by_value(816_127_900_000), [], id="a value off by more than the roundings"),
        pytest.param(
            by_value(816_127_900_000, tol=6e4),
            [("RevenueFromOperations", "consolidated", None)],
            id="within the label's rounding too",
        ),
        pytest.param(
            by_value(700_000_000_000),
            [("RevenueFromOperations", "standalone", None)],
            id="the other basis",
        ),
        pytest.param(
            by_value(880_906_800_000),
            [("Assets", "consolidated", None)],
            id="a balance at the year's end",
        ),
        pytest.param(
            by_value(358_935_700_000),
            [("SegmentRevenue", "consolidated", "FMCG - Cigarettes")],
            id="a segment",
        ),
        pytest.param(
            by_value(-37_068_000_000),
            [("PurchaseOfPropertyPlantAndEquipment", "consolidated", None)],
            id="a payment, sign aside",
        ),
        pytest.param(
            by_value(3.03, ("pure",)),
            [("CurrentRatio", "consolidated", None)],
            id="a ratio in its unit",
        ),
        pytest.param(by_value(3.03, ("INRPerShare",)), [], id="the same number in another unit"),
        pytest.param(by_value(31_000_000, tol=5e6), [], id="a fact rounded too coarsely to tell"),
        pytest.param(by_value(14.35, ("INRPerShare",)), [], id="another run's unchecked read"),
    ],
)
def test_a_figure_is_held_by_the_facts_that_match_it(
    conn: sqlite3.Connection, figure: Figure, held: list[tuple[str, str, str | None]]
) -> None:
    got = [(h.concept, h.basis, h.segment) for h in holdings(conn, figure)]
    assert got == held, got


def test_a_concept_is_held_from_its_earliest_filing(conn: sqlite3.Connection) -> None:
    with conn:
        row = FilingRow("ITC", 2026, date(2026, 3, 31), "consolidated", "xbrl", "u", "s", "t")
        restated = [_d("RevenueFromOperations", 900_000_000_000)]
        sync_xbrl_facts(
            conn, upsert_filing(conn, row), "ITC", "consolidated", restated, FACTS_WRITER
        )
    [held] = holdings(conn, by_concept("RevenueFromOperations", 1.0))
    assert held.value == 816_127_800_000


def test_holdings_put_consolidated_and_unsegmented_first(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            "UPDATE facts SET value = 816127800000 WHERE concept = 'RevenueFromOperations'"
        )
        conn.execute("UPDATE facts SET value = 816127800000 WHERE segment = 'FMCG - Cigarettes'")
    got = [(h.basis, h.segment) for h in holdings(conn, by_value(816_127_800_000))]
    assert got == [
        ("consolidated", None),
        ("consolidated", "FMCG - Cigarettes"),
        ("standalone", None),
    ]


# Found and exact.


@pytest.mark.parametrize(
    ("figure", "phrasing", "held", "rank", "exact"),
    [
        pytest.param(
            by_concept("RevenueFromOperations", 816_127_800_000),
            "asked",
            True,
            1,
            True,
            id="by concept, at rank 1, exact",
        ),
        pytest.param(
            by_concept("RevenueFromOperations", 816_127_900_000),
            "asked",
            True,
            1,
            False,
            id="by concept, a different value",
        ),
        pytest.param(
            by_concept("RevenueFromOperations", -816_127_800_000),
            "asked",
            True,
            1,
            False,
            id="by concept, the sign wrong",
        ),
        pytest.param(
            by_concept(None, 14.35), "asked", False, None, False, id="by concept, not held"
        ),
        pytest.param(
            by_value(358_935_700_000, asked="segment revenue"),
            "asked",
            True,
            1,
            None,
            id="by value, in its segment",
        ),
        pytest.param(
            by_value(700_000_000_000),
            "asked",
            True,
            1,
            None,
            id="by value, on the other basis",
        ),
        pytest.param(
            by_value(3.03, ("pure",), asked="current ratio"),
            "asked",
            True,
            1,
            None,
            id="by value, a ratio",
        ),
        pytest.param(by_value(31_000_000), "asked", False, None, None, id="by value, not held"),
    ],
)
def test_a_figure_is_found_at_a_rank_and_exact_or_not(
    conn: sqlite3.Connection,
    lookup: Lookup,
    figure: Figure,
    phrasing: Phrasing,
    held: bool,
    rank: int | None,
    exact: bool | None,
) -> None:
    got = score(conn, lookup, figure, phrasing)
    assert (got.held, got.rank, got.exact) == (held, rank, exact), got.detail


def test_a_held_figure_ranked_below_another_is_found_at_its_rank(
    conn: sqlite3.Connection, lookup: Lookup
) -> None:
    figure = by_value(880_906_800_000, asked="revenue from operations total assets")
    got = score(conn, lookup, figure, "asked")
    assert got.rank is not None and got.rank > 1, got.detail


class _Refusing(Lookup):
    def __call__(
        self,
        ticker: str,
        year: int,
        what: str,
        basis: Basis = "consolidated",
        period: Period = "year",
        segment: str | None = None,
    ) -> list[Found]:
        raise NotFound("nothing")


def test_a_held_figure_the_lookup_refuses_is_held_and_not_found(
    db: Path, conn: sqlite3.Connection
) -> None:
    refusing = _Refusing(db, concept_index(BagOfWordsEmbedder()), "r", "test")
    got = score(conn, refusing, by_concept("RevenueFromOperations", 816_127_800_000), "asked")
    assert (got.held, got.rank, got.exact) == (True, None, False)
    assert got.detail == "held as RevenueFromOperations (consolidated); got nothing"


def _lookups(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT run_id FROM lookups ORDER BY run_id")]


def _log_one(conn: sqlite3.Connection, run_id: str) -> None:
    with conn:
        conn.execute(
            """INSERT INTO lookups (run_id, asked_by, asked_at, ticker, basis, period, what,
                      concept, found)
               VALUES (?, 'x', 't', 'ITC', 'consolidated', '2025 year', 'w', NULL, 0)""",
            (run_id,),
        )


def test_evaluate_asks_each_figure_both_ways_and_keeps_none_of_its_lookups(
    conn: sqlite3.Connection, lookup: Lookup
) -> None:
    _log_one(conn, "an-agent-run")
    scored = evaluate(conn, lookup, [by_concept("RevenueFromOperations", 816_127_800_000)])
    assert [s.phrasing for s in scored] == ["asked", "word"]
    assert _lookups(conn) == ["an-agent-run"]


def test_evaluate_deletes_its_lookups_when_a_figure_fails(
    conn: sqlite3.Connection, lookup: Lookup
) -> None:
    broken = Figure(
        "b",
        "tables",
        "ITC",
        2025,
        816_127_800_000,
        0.0,
        {"asked": "revenue"},
        "value",
        units=("INR",),
    )
    with pytest.raises(KeyError):
        evaluate(conn, lookup, [broken])
    assert _lookups(conn) == []


# The summary and the report.


def _s(
    source: str,
    phrasing: Phrasing,
    held: bool,
    rank: int | None,
    exact: bool | None,
    id_: str = "x",
) -> Scored:
    return Scored(id_, source, phrasing, held, rank, exact, "why")


ROWS = [
    _s("fundamentals", "asked", True, 1, True, "ITC:2025:revenue"),
    _s("fundamentals", "asked", True, 3, False, "ITC:2025:eps"),
    _s("fundamentals", "asked", False, None, False, "ITC:2025:dividend"),
    _s("tables", "word", True, 2, None, "rt-001"),
    _s("tables", "word", True, None, None, "rt-002"),
    _s("teacher-results", "asked", False, None, None, "BAJAJ-AUTO-FY24#0004/0"),
]


def test_the_summary_counts_found_and_exact_out_of_the_held() -> None:
    lines = {(line.source, line.phrasing): line for line in summarise(ROWS)}
    assert list(lines) == [
        ("fundamentals", "asked"),
        ("tables", "word"),
        ("teacher-results", "asked"),
    ]
    first, last = COVERAGE_FOUND_AT[0], COVERAGE_FOUND_AT[-1]
    fundamentals = lines[("fundamentals", "asked")]
    assert (fundamentals.figures, fundamentals.held, fundamentals.exact) == (3, 2, 1)
    assert (fundamentals.found[first], fundamentals.found[last]) == (1, 2)
    tables = lines[("tables", "word")]
    assert (tables.held, tables.found[first], tables.found[last], tables.exact) == (2, 0, 1, None)


def test_the_report_heads_with_the_fundamentals_and_lists_their_and_the_tables_misses() -> None:
    left = [Left("rt-009", "tables", "a series"), Left("rt-020", "tables", "a series")]
    text = to_markdown(ROWS, left, {"database": "test"})
    headline = text.split("## Headline")[1].split("## Every source")[0]
    assert "| fundamentals | asked | 3 | 2/3 (67%) | 1/2 (50%) |" in headline
    assert "tables" not in headline
    assert "| tables | a series | 2 |" in text
    misses = text.split("## Misses")[1]
    assert "`ITC:2025:eps` [asked]" in misses and "`ITC:2025:dividend` [asked]" in misses
    assert "`rt-002` [word]" in misses and "`rt-001`" not in misses
    assert "`ITC:2025:revenue`" not in misses and "BAJAJ-AUTO" not in misses
    assert "- database: test" in text


def test_the_report_is_written_with_its_rows_beside_it(tmp_path: Path) -> None:
    path = write_report(ROWS, [], {}, datetime(2026, 10, 25, 14, 56), tmp_path)
    assert path.name == "coverage-20261025-1456.md"
    rows = path.with_suffix(".rows.jsonl").read_text().splitlines()
    assert [json.loads(r)["id"] for r in rows] == [r.id for r in ROWS]


def test_the_cli_runs_coverage_without_building(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[datetime] = []
    monkeypatch.setattr(cli, "coverage", ran.append)
    monkeypatch.setattr(cli, "build", lambda *a: pytest.fail("built"))
    cli.main(["coverage"])
    assert len(ran) == 1


def test_the_cli_scores_every_source_and_reports_the_database(
    monkeypatch: pytest.MonkeyPatch, db: Path, tmp_path: Path
) -> None:
    table = {("ITC", 2025, "revenue"): fundamental("ITC", 2025, "revenue", 816_127_800_000)}
    written: list[tuple[list[Scored], list[Left], dict[str, str]]] = []
    monkeypatch.setattr(cli, "FACTS_DB", db)
    monkeypatch.setattr(cli, "concept_index", lambda: concept_index(BagOfWordsEmbedder()))
    monkeypatch.setattr(cli, "load_fundamentals", lambda: table)
    monkeypatch.setattr(
        cli, "table_figures", lambda: ([by_value(3.03, ("pure",))], [Left("a", "tables", "r")])
    )
    monkeypatch.setattr(cli, "teacher_figures", lambda: ([], [Left("b", "teacher", "r")]))
    monkeypatch.setattr(
        cli, "write_report", lambda s, out, h, now: written.append((s, out, h)) or tmp_path / "r.md"
    )
    cli.coverage(datetime(2026, 10, 25, 15, 0))
    [(scored, left, header)] = written
    assert [(s.id, s.phrasing) for s in scored] == [
        ("ITC:2025:revenue", "asked"),
        ("ITC:2025:revenue", "word"),
        ("f", "asked"),
        ("f", "word"),
    ]
    assert [x.id for x in left] == ["a", "b"]
    assert header["database"] == "`facts.sqlite`, 2 filings, 10 facts"
    conn = connect(db)
    assert _lookups(conn) == []
    conn.close()
