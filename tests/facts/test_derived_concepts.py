"""`facts.derived_concepts`: offsets, plans (through derived concepts too) and their evaluation,
matching words strictly, the checks on a name and its bindings in a scope, the stored shape,
seeding, and the verify command."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from aigent.facts.derived_concepts import (
    At,
    Binding,
    DerivedConcept,
    Fetch,
    Step,
    as_bindings,
    available,
    bindings_json,
    bindings_refusal,
    comparing,
    evaluate,
    kind,
    load_derived_concepts,
    matched,
    name_refusal,
    plan,
    ranked,
    seed,
)
from aigent.facts.lookup import Period
from aigent.facts.store import Writer, connect

from .conftest import FIXTURES

WRITER = Writer("seed|test", "t", "seed-1")
AI = "derived-concept-ai"
GROWTH = DerivedConcept(
    "OperatingProfitGrowth",
    "banking",
    "Growth in operating profit",
    "growth",
    {
        "current": Binding("OperatingProfit", "current"),
        "previous": Binding("OperatingProfit", "previous"),
    },
    AI,
)
MARGIN = DerivedConcept(
    "NetProfitMargin",
    "banking",
    "Net profit over total income",
    "ratio",
    {"numerator": Binding("Profit", "current"), "denominator": Binding("Income", "current")},
    AI,
)
NII = DerivedConcept(
    "NetInterestIncome",
    "banking",
    "Interest earned less interest expended",
    "difference",
    {
        "minuend": Binding("InterestEarned", "current"),
        "subtrahend": Binding("InterestExpended", "current"),
    },
    AI,
)
NII_GROWTH = DerivedConcept(
    "NetInterestIncomeGrowth",
    "banking",
    "Growth in net interest income",
    "growth",
    {
        "current": Binding("NetInterestIncome", "current"),
        "previous": Binding("NetInterestIncome", "previous"),
    },
    AI,
)


@pytest.mark.parametrize(
    ("year", "period", "previous"),
    [
        pytest.param(2025, "year", (2024, "year"), id="a year over the year before"),
        pytest.param(2026, "Q1", (2025, "Q1"), id="a quarter over the same quarter a year before"),
        pytest.param(2026, "Q1", (2025, "Q4"), id="Q1 over the quarter before, last year's Q4"),
        pytest.param(2026, "Q3", (2026, "Q1"), id="a quarter over any earlier quarter"),
        pytest.param(2026, "H1", (2025, "H1"), id="a half over the half a year before"),
    ],
)
def test_a_growth_plan_reads_the_previous_period_it_is_given(
    year: int, period: Period, previous: tuple[int, Period]
) -> None:
    assert plan(GROWTH, year, period, previous=previous).fetches() == [
        Fetch("current", "OperatingProfit", year, period),
        Fetch("previous", "OperatingProfit", *previous),
    ]


@pytest.mark.parametrize(
    ("concept", "year", "period", "previous", "said"),
    [
        pytest.param(GROWTH, 2025, "year", None, "give the previous", id="growth with no previous"),
        pytest.param(MARGIN, 2025, "year", (2024, "year"), "one period", id="a ratio given one"),
        pytest.param(GROWTH, 2026, "Q1", (2025, "year"), "not as long", id="a quarter over a year"),
        pytest.param(GROWTH, 2026, "H1", (2025, "9M"), "not as long", id="a half over nine months"),
        pytest.param(GROWTH, 2025, "Q2", (2025, "Q3"), "does not end", id="previous after current"),
        pytest.param(GROWTH, 2025, "year", (2025, "year"), "does not end", id="the same period"),
    ],
)
def test_a_plan_refuses_periods_that_do_not_fit_the_metric(
    concept: DerivedConcept,
    year: int,
    period: Period,
    previous: tuple[int, Period] | None,
    said: str,
) -> None:
    with pytest.raises(ValueError, match=said):
        plan(concept, year, period, previous=previous)


def test_a_plan_refuses_an_input_that_compares_periods_itself() -> None:
    on_growth = DerivedConcept(
        "GrowthOfGrowth",
        "banking",
        "g",
        "growth",
        {"current": Binding(GROWTH.name, "current"), "previous": Binding(GROWTH.name, "previous")},
        AI,
    )
    with pytest.raises(ValueError, match="cannot be an input"):
        plan(on_growth, 2025, "year", {GROWTH.name: GROWTH}, (2024, "year"))


def test_a_plan_lists_each_input_in_formula_order_with_its_cite() -> None:
    fetches = plan(GROWTH, 2025, "year", previous=(2024, "year")).fetches()
    assert fetches == [
        Fetch("current", "OperatingProfit", 2025, "year"),
        Fetch("previous", "OperatingProfit", 2024, "year"),
    ]
    assert fetches[1].cite_as("ICICIBANK", "standalone") == (
        "FACT:ICICIBANK:2024:standalone:year:OperatingProfit"
    )


def test_a_plan_runs_through_a_derived_input_at_its_own_period() -> None:
    step = plan(NII_GROWTH, 2025, "year", {"NetInterestIncome": NII}, (2024, "year"))
    assert step.fetches() == [
        Fetch("current.minuend", "InterestEarned", 2025, "year"),
        Fetch("current.subtrahend", "InterestExpended", 2025, "year"),
        Fetch("previous.minuend", "InterestEarned", 2024, "year"),
        Fetch("previous.subtrahend", "InterestExpended", 2024, "year"),
    ]
    assert isinstance(step.inputs["previous"], Step) and step.inputs["previous"].year == 2024
    assert step.previous == (2024, "year")


def test_a_nested_plan_evaluates_bottom_up() -> None:
    step = plan(NII_GROWTH, 2025, "year", {"NetInterestIncome": NII}, (2024, "year"))
    facts = {
        "current.minuend": (150.0, "INR"),
        "current.subtrahend": (30.0, "INR"),
        "previous.minuend": (130.0, "INR"),
        "previous.subtrahend": (30.0, "INR"),
    }
    assert evaluate(step, facts) == (pytest.approx(0.2), "pure")


@pytest.mark.parametrize(
    ("units", "said"),
    [
        pytest.param(("INR", "INR"), None, id="one unit"),
        pytest.param(("INR", "INRPerShare"), "needs one unit", id="two units"),
    ],
)
def test_a_same_unit_formula_refuses_mixed_units(units: tuple[str, str], said: str | None) -> None:
    facts = {"minuend": (5.0, units[0]), "subtrahend": (2.0, units[1])}
    if said is None:
        assert evaluate(plan(NII, 2025, "year"), facts) == (3.0, "INR")
    else:
        with pytest.raises(ValueError, match=said):
            evaluate(plan(NII, 2025, "year"), facts)


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
        pytest.param("the dividend", [], id="nothing shared but filler"),
    ],
)
def test_ranked(what: str, names: list[str]) -> None:
    assert [m.name for m in ranked(what, [GROWTH, MARGIN])] == names


@pytest.mark.parametrize(
    ("what", "name"),
    [
        pytest.param("NetProfitMargin", "NetProfitMargin", id="an exact name"),
        pytest.param("operating profit growth", "OperatingProfitGrowth", id="every word held"),
        pytest.param(
            "growth in the operating profit", "OperatingProfitGrowth", id="filler skipped"
        ),
        pytest.param("basic earnings per share growth", None, id="only growth shared"),
        pytest.param("ratio of other income to total income", None, id="a near miss"),
        pytest.param("of the", None, id="filler only"),
    ],
)
def test_matched_plans_only_a_metric_holding_every_word(what: str, name: str | None) -> None:
    found = matched(what, [GROWTH, MARGIN])
    assert (found.name if found else None) == name


@pytest.mark.parametrize(
    ("name", "said"),
    [
        pytest.param("CostToIncomeRatio", None, id="CamelCase"),
        pytest.param("OperatingProfitBeforeProvisionAndContingenciesGrowth", None, id="a long one"),
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
            "ratio", {"numerator": Binding("Profit", "current")}, "takes", id="a variable missing"
        ),
        pytest.param(
            "ratio",
            {**MARGIN.bindings, "numerator": Binding("Dividend", "current")},
            "no concept Dividend",
            id="a concept the scope does not file",
        ),
        pytest.param(
            "ratio",
            {**MARGIN.bindings, "numerator": Binding("Profit", cast(At, "t-1y"))},
            "at t-1y",
            id="an offset, from before periods were named when planned",
        ),
        pytest.param(
            "ratio",
            {**MARGIN.bindings, "numerator": Binding("ProfitGrowth", "current")},
            "compares two periods itself",
            id="an input comparing two periods",
        ),
        pytest.param(
            "ratio",
            {**MARGIN.bindings, "numerator": Binding("NetProfitMargin", "current")},
            "input to itself",
            id="itself as an input",
        ),
    ],
)
def test_bindings_refusal(formula: str, bindings: dict[str, Binding], said: str | None) -> None:
    why = bindings_refusal(
        formula,
        bindings,
        CONCEPTS | {"NetProfitMargin", "ProfitGrowth"},
        "NetProfitMargin",
        {"ProfitGrowth"},
    )
    assert (why is None) if said is None else (why is not None and said in why), why


def test_bindings_round_trip_with_variables_sorted() -> None:
    stored = bindings_json(GROWTH.bindings)
    assert list(json.loads(stored)) == ["current", "previous"]
    assert as_bindings(stored) == GROWTH.bindings


def _filed(conn: sqlite3.Connection, ticker: str, *concepts: str) -> None:
    for concept in concepts:
        conn.execute(
            """INSERT INTO facts (ticker, basis, concept, label, period_kind, period_end, value,
                      unit, tier, written_by, written_at, run_id)
               VALUES (?, 'standalone', ?, ?, 'instant', '2025-03-31', 1, 'INR', 'xbrl-ai', 'w',
                       't', 'r')""",
            (ticker, concept, concept),
        )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    """A bank filing Profit and Income, and a company filing ProfitBeforeTax."""
    conn = connect(tmp_path / "facts.sqlite")
    with conn:
        _filed(conn, "ICICIBANK", "Profit", "Income")
        _filed(conn, "ITC", "ProfitBeforeTax")
    return conn


def test_available_is_what_a_scope_files_and_derives(conn: sqlite3.Connection) -> None:
    assert available(conn, "banking") == {"Profit", "Income"}
    assert available(conn, "ind-as") == {"ProfitBeforeTax"}
    conn.execute(
        """INSERT INTO derived_concepts (name, scope, label, formula, bindings, tier, written_by,
                  written_at, run_id)
           VALUES ('M', 'banking', 'm', 'ratio', '{}', ?, 'w', 't', 'r')""",
        (AI,),
    )
    assert available(conn, "banking") == {"Profit", "Income", "M"}
    assert "M" not in available(conn, "ind-as")


def _seeds(tmp_path: Path, *rows: dict[str, object]) -> Path:
    path = tmp_path / "derived-concepts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


MARGIN_ROW: dict[str, object] = {
    "name": "NetProfitMargin",
    "scope": "banking",
    "label": "Net profit over total income",
    "formula": "ratio",
    "bindings": {
        "denominator": {"concept": "Income", "at": "current"},
        "numerator": {"concept": "Profit", "at": "current"},
    },
}
ON_MARGIN: dict[str, object] = {
    "name": "NetProfitMarginGrowth",
    "scope": "banking",
    "label": "Growth in net profit margin",
    "formula": "growth",
    "bindings": {
        "current": {"concept": "NetProfitMargin", "at": "current"},
        "previous": {"concept": "NetProfitMargin", "at": "previous"},
    },
}


def test_a_kind_is_read_through_derived_inputs(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """INSERT INTO facts (ticker, basis, concept, label, period_kind, period_start,
                      period_end, value, unit, tier, written_by, written_at, run_id)
               VALUES ('ICICIBANK', 'standalone', 'Expense', 'e', 'duration', '2024-04-01',
                       '2025-03-31', 1, 'INR', 'xbrl-ai', 'w', 't', 'r')"""
        )
    on_margin = DerivedConcept(
        "MarginShare",
        "banking",
        "m",
        "ratio",
        {"numerator": Binding(MARGIN.name, "current"), "denominator": Binding("Income", "current")},
        AI,
    )
    over_flow = DerivedConcept(
        "MarginOverExpense",
        "banking",
        "m",
        "ratio",
        {
            "numerator": Binding(MARGIN.name, "current"),
            "denominator": Binding("Expense", "current"),
        },
        AI,
    )
    held = {MARGIN.name: MARGIN}
    assert kind(conn, on_margin, held) == "snapshot", "balances all the way down"
    assert kind(conn, over_flow, held) == "mixed"
    assert kind(conn, GROWTH, {}) == "flow", "a concept with no facts reads as a flow"


def test_a_growth_is_comparing_in_its_scope_and_no_input_to_another(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    growth = {
        **MARGIN_ROW,
        "name": "ProfitGrowth",
        "formula": "growth",
        "bindings": {
            "current": {"at": "current", "concept": "Profit"},
            "previous": {"at": "previous", "concept": "Profit"},
        },
    }
    on_growth = {
        **MARGIN_ROW,
        "name": "ProfitGrowthShare",
        "bindings": {
            "denominator": {"at": "current", "concept": "Income"},
            "numerator": {"at": "current", "concept": "ProfitGrowth"},
        },
    }
    refused = seed(conn, WRITER, _seeds(tmp_path, growth, MARGIN_ROW, on_growth))
    assert refused == [
        "ProfitGrowthShare [banking]: ProfitGrowth compares two periods itself; it cannot be "
        "an input"
    ]
    assert comparing(conn, "banking") == {"ProfitGrowth"}
    assert comparing(conn, "ind-as") == set()


def test_seed_writes_each_once_in_any_order_and_reports_the_refused(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    bad = {
        **MARGIN_ROW,
        "name": "DividendGrowth",
        "bindings": {"x": {"concept": "D", "at": "current"}},
    }
    path = _seeds(tmp_path, ON_MARGIN, MARGIN_ROW, bad)
    refused = ["DividendGrowth [banking]: ratio takes denominator, numerator, not x"]
    assert seed(conn, WRITER, path) == refused
    assert seed(conn, WRITER, path) == refused
    held = load_derived_concepts(conn)
    assert [m.name for m in held] == ["NetProfitMargin", "NetProfitMarginGrowth"], (
        "one built on one"
    )
    row = conn.execute("SELECT tier, written_by, question FROM derived_concepts").fetchone()
    assert row == (AI, "seed|test", None)
    assert held[0] == DerivedConcept(
        "NetProfitMargin", "banking", "Net profit over total income", "ratio", MARGIN.bindings, AI
    )


@pytest.mark.parametrize(
    ("changed", "said"),
    [
        pytest.param({"name": "NetProfitMarginFY25"}, "names a period", id="a period in the name"),
        pytest.param({"scope": "insurance"}, "scope is", id="an unknown scope"),
        pytest.param({"scope": "banking,*"}, "scope is", id="star in a list"),
        pytest.param({"scope": "ind-as"}, "no concept", id="a concept its scope does not file"),
    ],
)
def test_seed_refuses_a_bad_row(
    conn: sqlite3.Connection, tmp_path: Path, changed: dict[str, object], said: str
) -> None:
    assert said in seed(conn, WRITER, _seeds(tmp_path, {**MARGIN_ROW, **changed}))[0]
    assert load_derived_concepts(conn) == []


def test_a_name_may_be_held_once_per_scope(conn: sqlite3.Connection, tmp_path: Path) -> None:
    other = {
        **MARGIN_ROW,
        "scope": "ind-as",
        "bindings": {
            "denominator": {"concept": "ProfitBeforeTax", "at": "current"},
            "numerator": {"concept": "ProfitBeforeTax", "at": "current"},
        },
    }
    assert seed(conn, WRITER, _seeds(tmp_path, MARGIN_ROW, other)) == []
    assert [(m.name, m.scope) for m in load_derived_concepts(conn)] == [
        ("NetProfitMargin", "banking"),
        ("NetProfitMargin", "ind-as"),
    ]
    assert [m.scope for m in load_derived_concepts(conn, "ind-as")] == ["ind-as"]


def test_the_seeded_metrics_are_all_well_formed() -> None:
    from aigent.facts.derived_concepts import SEEDED
    from aigent.facts.formulas import FORMULAS

    for line in SEEDED.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert name_refusal(row["name"]) is None, row["name"]
        bindings = as_bindings(json.dumps(row["bindings"]))
        concepts = {b.concept for b in bindings.values()}
        assert bindings_refusal(row["formula"], bindings, concepts, row["name"]) is None
        assert set(bindings) == set(FORMULAS[row["formula"]].variables)


def _shipped_ind_as(conn: sqlite3.Connection, tmp_path: Path) -> list[str]:
    """Seeds the shipped ind-as rows over the concepts every ind-as company files; the refused."""
    from aigent.facts.derived_concepts import SEEDED

    filed = (FIXTURES / "ind-as-filed-concepts.txt").read_text(encoding="utf-8").split()
    with conn:
        _filed(conn, "WIPRO", *filed)
    rows = [json.loads(line) for line in SEEDED.read_text(encoding="utf-8").splitlines()]
    return seed(conn, WRITER, _seeds(tmp_path, *(r for r in rows if r["scope"] == "ind-as")))


def test_the_shipped_ind_as_seeds_rest_on_what_every_ind_as_company_files(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "facts.sqlite")
    assert _shipped_ind_as(conn, tmp_path) == []
    assert "EBITDAMargin" in {m.name for m in load_derived_concepts(conn, "ind-as")}


def test_the_shipped_ebitda_margin_excludes_finance_costs_and_depreciation(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "facts.sqlite")
    _shipped_ind_as(conn, tmp_path)
    held = {m.name: m for m in load_derived_concepts(conn, "ind-as")}
    step = plan(held["EBITDAMargin"], 2025, "year", held)
    given = {
        "RevenueFromOperations": 200.0,
        "Expenses": 180.0,
        "FinanceCosts": 10.0,
        "DepreciationDepletionAndAmortisationExpense": 20.0,
    }
    facts = {f.variable: (given[f.concept], "INR") for f in step.fetches()}
    assert evaluate(step, facts) == (pytest.approx(0.25), "pure"), facts


def test_an_exact_name_comes_first_where_shared_words_tie() -> None:
    ratio = DerivedConcept("Ratio", "banking", "a ratio", "ratio", MARGIN.bindings, AI)
    credit = DerivedConcept("CreditRatio", "banking", "credit over deposits", "ratio", {}, AI)
    assert [m.name for m in ranked("Ratio", [credit, ratio])] == ["Ratio", "CreditRatio"]


def test_the_command_shows_a_concept_and_records_the_check(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from aigent.facts import __main__ as cli

    seed(conn, WRITER, _seeds(tmp_path, MARGIN_ROW))
    db = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    conn.close()
    monkeypatch.setattr(cli, "FACTS_DB", db)
    ask = ["verify-concept", "--name", "NetProfitMargin", "--scope", "banking"]
    cli.main([*ask, "--by", "vedant"])
    out = capsys.readouterr().out
    assert "[banking]" in out and "was derived-concept-ai" in out and "by vedant" in out, out
    with pytest.raises(SystemExit, match="no derived concept"):
        cli.main(["verify-concept", "--name", "NetProfitMargin", "--scope", "ind-as", "--by", "v"])
    with pytest.raises(SystemExit, match="who checked"):
        cli.main([*ask, "--by", ""])
