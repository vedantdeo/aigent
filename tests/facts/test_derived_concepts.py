"""`facts.derived_concepts`: offsets, plans, matching words to derived concepts, the checks on a
name and its bindings, the stored shape, and seeding."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from aigent.facts.derived_concepts import (
    Binding,
    DerivedConcept,
    Fetch,
    Offset,
    as_bindings,
    bindings_json,
    bindings_refusal,
    load_derived_concepts,
    name_refusal,
    plan,
    ranked,
    seed,
    shift,
)
from aigent.facts.lookup import Period
from aigent.facts.store import Writer, connect

WRITER = Writer("seed|test", "t", "seed-1")
GROWTH = DerivedConcept(
    "OperatingProfitGrowth",
    "Growth in operating profit",
    "growth",
    {"current": Binding("OperatingProfit", "t"), "previous": Binding("OperatingProfit", "t-1y")},
    "derived-concept-ai",
)
MARGIN = DerivedConcept(
    "NetProfitMargin",
    "Net profit over total income",
    "ratio",
    {"numerator": Binding("Profit", "t"), "denominator": Binding("Income", "t")},
    "derived-concept-ai",
)


@pytest.mark.parametrize(
    ("year", "period", "offset", "expected"),
    [
        pytest.param(2025, "year", "t", (2025, "year"), id="the period asked"),
        pytest.param(2025, "year", "t-1y", (2024, "year"), id="a year before"),
        pytest.param(2025, "Q3", "t-1y", (2024, "Q3"), id="the same quarter a year before"),
        pytest.param(2025, "Q3", "t-1q", (2025, "Q2"), id="the quarter before"),
        pytest.param(2025, "Q1", "t-1q", (2024, "Q4"), id="Q1's quarter before is last year's Q4"),
    ],
)
def test_shift(year: int, period: Period, offset: Offset, expected: tuple[int, str]) -> None:
    assert shift(year, period, offset) == expected


@pytest.mark.parametrize("period", ["year", "H1", "9M"])
def test_a_quarter_before_a_period_that_is_no_quarter_is_refused(period: Period) -> None:
    with pytest.raises(ValueError, match="not a quarter"):
        shift(2025, period, "t-1q")


def test_a_plan_lists_each_input_in_formula_order_with_its_cite() -> None:
    fetches = plan(GROWTH, 2025, "year")
    assert fetches == [
        Fetch("current", "OperatingProfit", 2025, "year"),
        Fetch("previous", "OperatingProfit", 2024, "year"),
    ]
    assert fetches[1].cite_as("ICICIBANK", "standalone") == (
        "FACT:ICICIBANK:2024:standalone:year:OperatingProfit"
    )


@pytest.mark.parametrize(
    ("what", "names"),
    [
        pytest.param(
            "operating profit growth",
            ["OperatingProfitGrowth", "NetProfitMargin"],
            id="the most words shared first",
        ),
        pytest.param(
            "NetProfitMargin", ["NetProfitMargin", "OperatingProfitGrowth"], id="an exact name"
        ),
        pytest.param(
            "net profit", ["NetProfitMargin", "OperatingProfitGrowth"], id="ties by shared count"
        ),
        pytest.param("dividend", [], id="nothing shared"),
    ],
)
def test_ranked(what: str, names: list[str]) -> None:
    assert [m.name for m in ranked(what, [GROWTH, MARGIN])] == names


@pytest.mark.parametrize(
    ("name", "said"),
    [
        pytest.param("CostToIncomeRatio", None, id="CamelCase"),
        pytest.param("ProfitLossForThePeriodQuarterOnQuarterGrowth", None, id="a long one"),
        pytest.param("cost to income", "not a CamelCase", id="words"),
        pytest.param("costToIncome", "not a CamelCase", id="lower first"),
        pytest.param("ProfitGrowthFY25", "names a period", id="a financial year"),
        pytest.param("Profit2025", "names a period", id="a calendar year"),
        pytest.param("ProfitQ3", "names a period", id="a quarter"),
    ],
)
def test_name_refusal(name: str, said: str | None) -> None:
    why = name_refusal(name)
    assert (why is None) if said is None else (why is not None and said in why), why


CONCEPTS = {"Profit", "Income"}


@pytest.mark.parametrize(
    ("formula", "bindings", "said"),
    [
        pytest.param("ratio", MARGIN.bindings, None, id="good"),
        pytest.param("mean", MARGIN.bindings, "no formula", id="an unknown formula"),
        pytest.param(
            "ratio", {"numerator": Binding("Profit", "t")}, "takes", id="a variable missing"
        ),
        pytest.param(
            "ratio",
            {**MARGIN.bindings, "numerator": Binding("Dividend", "t")},
            "no concept Dividend",
            id="an unknown concept",
        ),
        pytest.param(
            "ratio",
            {**MARGIN.bindings, "numerator": Binding("Profit", cast(Offset, "t-2y"))},
            "offset t-2y",
            id="an unknown offset",
        ),
    ],
)
def test_bindings_refusal(formula: str, bindings: dict[str, Binding], said: str | None) -> None:
    why = bindings_refusal(formula, bindings, CONCEPTS)
    assert (why is None) if said is None else (why is not None and said in why), why


def test_bindings_round_trip_with_variables_sorted() -> None:
    stored = bindings_json(GROWTH.bindings)
    assert list(json.loads(stored)) == ["current", "previous"]
    assert as_bindings(stored) == GROWTH.bindings


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "facts.sqlite")
    with conn:
        for concept in ("Profit", "Income"):
            conn.execute(
                "INSERT INTO concepts (concept, label, kind) VALUES (?, ?, 'xbrl')",
                (concept, concept),
            )
    return conn


def _seeds(tmp_path: Path, *rows: dict[str, object]) -> Path:
    path = tmp_path / "derived-concepts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


MARGIN_ROW = {
    "name": "NetProfitMargin",
    "label": "Net profit over total income",
    "formula": "ratio",
    "bindings": {
        "denominator": {"concept": "Income", "offset": "t"},
        "numerator": {"concept": "Profit", "offset": "t"},
    },
}


def test_seed_writes_each_metric_once_and_reports_the_refused(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    bad = {
        **MARGIN_ROW,
        "name": "DividendGrowth",
        "bindings": {"x": {"concept": "D", "offset": "t"}},
    }
    path = _seeds(tmp_path, MARGIN_ROW, bad)
    assert seed(conn, WRITER, path) == ["DividendGrowth: ratio takes denominator, numerator, not x"]
    assert seed(conn, WRITER, path) == ["DividendGrowth: ratio takes denominator, numerator, not x"]
    assert [m.name for m in load_derived_concepts(conn)] == ["NetProfitMargin"], "written once"
    row = conn.execute("SELECT tier, written_by, question FROM derived_concepts").fetchone()
    assert row == ("derived-concept-ai", "seed|test", None)
    assert load_derived_concepts(conn)[0] == DerivedConcept(
        "NetProfitMargin",
        "Net profit over total income",
        "ratio",
        MARGIN.bindings,
        "derived-concept-ai",
    )


def test_seed_refuses_a_name_with_a_period(conn: sqlite3.Connection, tmp_path: Path) -> None:
    path = _seeds(tmp_path, {**MARGIN_ROW, "name": "NetProfitMarginFY25"})
    assert "names a period" in seed(conn, WRITER, path)[0]
    assert load_derived_concepts(conn) == []


def test_the_seeded_metrics_are_all_well_formed() -> None:
    from aigent.facts.derived_concepts import SEEDED
    from aigent.facts.formulas import FORMULAS

    for line in SEEDED.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert name_refusal(row["name"]) is None, row["name"]
        bindings = as_bindings(json.dumps(row["bindings"]))
        concepts = {b.concept for b in bindings.values()}
        assert bindings_refusal(row["formula"], bindings, concepts) is None, row["name"]
        assert set(bindings) == set(FORMULAS[row["formula"]].variables)


def test_an_exact_name_comes_first_where_shared_words_tie() -> None:
    ratio = DerivedConcept("Ratio", "a ratio", "ratio", MARGIN.bindings, "derived-concept-ai")
    credit = DerivedConcept(
        "CreditRatio", "credit over deposits", "ratio", MARGIN.bindings, "derived-concept-ai"
    )
    assert [m.name for m in ranked("Ratio", [credit, ratio])] == ["Ratio", "CreditRatio"]
