"""`analyst.metric_tools`: a plan's fetch list, a computation checked against its plan, a metric
defined and a formula proposed, every refusal as an error result, and the write-back of what a
supported claim cites."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal, cast

import pytest
from anthropic.types import ToolParam

pytest.importorskip("langgraph")

from aigent.analyst.fundamentals import (  # noqa: E402
    COMPARE_TOOL,
    FUNDAMENTALS_TOOL,
    GET_METRIC_TOOL,
    FundamentalsTool,  # noqa: E402
)
from aigent.analyst.graph import Analysed  # noqa: E402
from aigent.analyst.memo import Claim, Judged, Memo  # noqa: E402
from aigent.analyst.metric_tools import (  # noqa: E402
    COMPUTE,
    DEFINE,
    METRIC_TOOLS,
    PLAN,
    PROPOSE,
    MetricTools,
    chained,
    write_back,
)
from aigent.facts.lookup import Lookup, concept_index  # noqa: E402
from aigent.facts.store import Writer, verify_derived_concept  # noqa: E402

from ..conftest import BagOfWordsEmbedder  # noqa: E402
from .conftest import ADV25, DEP25, EPS25, OP24, OP25, UNVERIFIED, evidence_of  # noqa: E402

# A snapshot planned as at Q4's end is fetched as at Q4, not over the year.
ADV25_Q4 = {**ADV25, "cite_as": str(ADV25["cite_as"]).replace(":year:", ":Q4:")}
DEP25_Q4 = {**DEP25, "cite_as": str(DEP25["cite_as"]).replace(":year:", ":Q4:")}

WRITER = Writer("analyst-desk|m", "2026-11-03T22:00:00+05:30", "desk-1")
QUESTION = "How much did ICICI Bank's operating profit grow in FY25?"
ASK: dict[str, object] = {
    "name": "OperatingProfitGrowth",
    "company": "ICICIBANK",
    "year": "2025",
    "quarter": "Q4",
    "period": "12M",
    "previous_year": "2024",
    "previous_quarter": "Q4",
    "previous_period": "12M",
    "basis": "standalone",
}
ONE: dict[str, object] = {"previous_year": None, "previous_quarter": None, "previous_period": None}
SNAP: dict[str, object] = {**ONE, "period": None}
Verdict = Literal["supported", "unsupported", "wrong"]


def _tools(conn: sqlite3.Connection, *facts: dict[str, object]) -> MetricTools:
    return MetricTools(conn, evidence_of(*facts), WRITER, QUESTION)


def _call(tools: MetricTools, name: str, arguments: dict[str, object]) -> dict[str, object]:
    content, is_error = tools(name, arguments)
    assert not is_error, content
    return json.loads(content)


def _computed(tools: MetricTools) -> dict[str, object]:
    planned = _call(tools, PLAN, ASK)
    fetch = cast(list[dict[str, object]], planned["fetch"])
    inputs = [{"variable": f["variable"], "cite_as": f["cite_as"]} for f in fetch]
    return _call(tools, COMPUTE, {"plan_id": planned["plan_id"], "inputs": inputs})


def test_a_plan_names_the_metric_its_formula_and_each_fact_to_fetch(
    concepts_db: sqlite3.Connection,
) -> None:
    planned = _call(_tools(concepts_db), PLAN, ASK)
    assert planned["plan_id"] == "P1" and planned["metric"] == "OperatingProfitGrowth"
    assert planned["tier"] == "derived-concept-ai" and str(planned["formula"]).startswith("growth:")
    assert planned["periods"] == "FY2025 over FY2024" and planned["kind"] == "flow"
    assert planned["fetch"] == [
        {
            "variable": "current",
            "fundamentals": {
                "company": "ICICIBANK",
                "concept": "OperatingProfit",
                "year": "2025",
                "quarter": "Q4",
                "period": "12M",
                "basis": "standalone",
                "segment": None,
            },
            "cite_as": OP25["cite_as"],
        },
        {
            "variable": "previous",
            "fundamentals": {
                "company": "ICICIBANK",
                "concept": "OperatingProfit",
                "year": "2024",
                "quarter": "Q4",
                "period": "12M",
                "basis": "standalone",
                "segment": None,
            },
            "cite_as": OP24["cite_as"],
        },
    ]


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"name": "DividendPayout"}, id="a name not held"),
        pytest.param({"name": "operating profit growth"}, id="words, not a name"),
        pytest.param({"company": "ITC"}, id="a name held in another scope"),
    ],
)
def test_a_name_not_held_points_to_get_metric_and_the_library(
    concepts_db: sqlite3.Connection, changed: dict[str, object]
) -> None:
    content, is_error = _tools(concepts_db)(PLAN, {**ASK, **changed})
    assert is_error and "get_metric" in content, content
    assert "growth(current, previous)" in content, content


@pytest.mark.parametrize(
    ("changed", "said"),
    [
        pytest.param({"name": " "}, "give the name", id="no name"),
        pytest.param({"company": "ACME"}, "give the name", id="a company not held"),
        pytest.param({"year": "FY25"}, "give the name", id="a year not in digits"),
        pytest.param({"quarter": "Q5"}, "quarter is one of", id="an unknown quarter"),
        pytest.param({"basis": "group"}, "quarter is one of", id="an unknown basis"),
        pytest.param({"period": "4M"}, "period is one of", id="an unknown period"),
        pytest.param(
            {
                "year": "2026",
                "quarter": "Q2",
                "period": "6M",
                "previous_year": "2025",
                "previous_quarter": "Q2",
                "previous_period": "6M",
            },
            None,
            id="six months over the six months a year before",
        ),
        pytest.param(
            {
                "year": "2026",
                "quarter": "Q1",
                "period": "3M",
                "previous_year": "2025",
                "previous_quarter": "Q4",
                "previous_period": "3M",
            },
            None,
            id="a quarter over the one before",
        ),
        pytest.param(ONE, "give the previous period", id="growth with no previous time"),
        pytest.param({"previous_year": None}, "null for both", id="a previous quarter, no year"),
        pytest.param({"previous_quarter": "Q5"}, "null for both", id="an unknown previous quarter"),
        pytest.param({"previous_year": "FY24"}, "null for both", id="a previous year not digits"),
        pytest.param(
            {"previous_year": None, "previous_quarter": None},
            "with a previous time",
            id="a previous period with no previous time",
        ),
        pytest.param({"period": None}, "reads a span", id="a flow with no period"),
        pytest.param({"quarter": "Q3"}, "ends at Q4 only", id="twelve months ending Q3"),
        pytest.param(
            {"previous_period": "3M", "previous_quarter": "Q3"},
            "as long as period",
            id="twelve months over three",
        ),
        pytest.param({"previous_year": "2025"}, "does not end", id="the same year"),
        pytest.param(
            {"name": "CreditDepositRatio", **ONE}, "a snapshot", id="a snapshot given a period"
        ),
        pytest.param(
            {"name": "CreditDepositRatio"}, "a snapshot", id="a snapshot given previous periods"
        ),
        pytest.param({"name": "CreditDepositRatio", **SNAP}, None, id="a snapshot at a time"),
    ],
)
def test_plan_refuses_a_bad_ask(
    concepts_db: sqlite3.Connection, changed: dict[str, object], said: str | None
) -> None:
    content, is_error = _tools(concepts_db)(PLAN, {**ASK, **changed})
    assert is_error is (said is not None) and (said is None or said in content), content


@pytest.mark.parametrize(
    ("asked", "shape", "periods"),
    [
        pytest.param(ASK, "flow", "FY2025 over FY2024", id="growth of a flow"),
        pytest.param(
            {**ASK, **SNAP, "name": "CreditDepositRatio"},
            "snapshot",
            "as at the end of Q4 FY2025",
            id="a ratio of balances",
        ),
        pytest.param(
            {**ASK, **ONE, "name": "OperatingProfitLessEps", "quarter": "Q1", "period": "3M"},
            "flow",
            "Q1 FY2025",
            id="a flow over one quarter",
        ),
    ],
)
def test_a_plan_says_its_kind_and_its_time(
    concepts_db: sqlite3.Connection, asked: dict[str, object], shape: str, periods: str
) -> None:
    planned = _call(_tools(concepts_db), PLAN, asked)
    assert (planned["kind"], planned["periods"]) == (shape, periods), planned


@pytest.mark.parametrize(
    "tool",
    [*METRIC_TOOLS, FUNDAMENTALS_TOOL, GET_METRIC_TOOL, COMPARE_TOOL],
    ids=lambda t: str(t["name"]),
)
def test_no_property_pairs_an_enum_with_a_list_of_types(tool: ToolParam) -> None:
    schema = cast(dict[str, object], tool["input_schema"])
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    paired = [n for n, p in properties.items() if "enum" in p and isinstance(p.get("type"), list)]
    assert paired == [], "the API refuses an enum under a list of types; use anyOf"


def test_a_computation_takes_the_values_from_the_planned_facts(
    concepts_db: sqlite3.Connection,
) -> None:
    tools = _tools(concepts_db, OP25, OP24)
    shown = _computed(tools)
    assert shown["cite_as"] == "DERIVED:D1" and shown["metric"] == "OperatingProfitGrowth"
    assert shown["value"] == pytest.approx((672988200000 - 581307600000) / 581307600000)
    inputs = cast(list[dict[str, object]], shown["inputs"])
    assert shown["unit"] == "pure" and [i["variable"] for i in inputs] == ["current", "previous"]
    assert json.loads(tools.evidence["DERIVED:D1"].text) == shown, "the critic reads it"


def test_a_ratio_of_balances_and_a_difference_keep_their_units(
    concepts_db: sqlite3.Connection,
) -> None:
    tools = _tools(concepts_db, ADV25_Q4, DEP25_Q4)
    planned = _call(tools, PLAN, {**ASK, **SNAP, "name": "CreditDepositRatio"})
    shown = _call(
        tools,
        COMPUTE,
        {
            "plan_id": planned["plan_id"],
            "inputs": [
                {"variable": "numerator", "cite_as": ADV25_Q4["cite_as"]},
                {"variable": "denominator", "cite_as": DEP25_Q4["cite_as"]},
            ],
        },
    )
    assert (shown["value"], shown["unit"]) == (0.8, "pure")


def _compute_with(
    conn: sqlite3.Connection, inputs: list[dict[str, object]], *facts: dict[str, object]
) -> tuple[str, bool]:
    tools = _tools(conn, *facts)
    planned = _call(tools, PLAN, ASK)
    return tools(COMPUTE, {"plan_id": planned["plan_id"], "inputs": inputs})


GOOD = [
    {"variable": "current", "cite_as": OP25["cite_as"]},
    {"variable": "previous", "cite_as": OP24["cite_as"]},
]


@pytest.mark.parametrize(
    ("inputs", "facts", "said"),
    [
        pytest.param(GOOD[:1], (OP25, OP24), "one cite_as for each", id="a variable missing"),
        pytest.param(GOOD + GOOD[:1], (OP25, OP24), "given twice", id="a variable twice"),
        pytest.param(
            [GOOD[0], {"variable": "previous", "cite_as": OP25["cite_as"]}],
            (OP25, OP24),
            "previous is",
            id="the wrong period's fact",
        ),
        pytest.param(GOOD, (OP25,), "not returned by fundamentals", id="a fact not fetched"),
        pytest.param(
            [GOOD[0], {"variable": "previous", "cite_as": UNVERIFIED["cite_as"]}],
            (OP25, OP24, UNVERIFIED),
            "previous is",
            id="another fact in its place",
        ),
    ],
)
def test_compute_refuses_what_its_plan_did_not_ask_for(
    concepts_db: sqlite3.Connection,
    inputs: list[dict[str, object]],
    facts: tuple[dict[str, object], ...],
    said: str,
) -> None:
    content, is_error = _compute_with(concepts_db, inputs, *facts)
    assert is_error and said in content, content


def test_compute_refuses_an_unverified_fact_even_when_planned(
    concepts_db: sqlite3.Connection,
) -> None:
    unchecked = {**OP24, "tier": "xbrl-ai"}
    content, is_error = _compute_with(concepts_db, GOOD, OP25, unchecked)
    assert is_error and "not verified" in content, content


def test_compute_refuses_an_unknown_plan_and_mixed_units(concepts_db: sqlite3.Connection) -> None:
    tools = _tools(concepts_db, OP25, EPS25)
    content, is_error = tools(COMPUTE, {"plan_id": "P9", "inputs": GOOD})
    assert is_error and "no plan 'P9'" in content, content
    planned = _call(tools, PLAN, {**ASK, **ONE, "name": "OperatingProfitLessEps"})
    content, is_error = tools(
        COMPUTE,
        {
            "plan_id": planned["plan_id"],
            "inputs": [
                {"variable": "minuend", "cite_as": OP25["cite_as"]},
                {"variable": "subtrahend", "cite_as": EPS25["cite_as"]},
            ],
        },
    )
    assert is_error and "needs one unit" in content, content


def test_compute_turns_a_zero_base_into_an_error(concepts_db: sqlite3.Connection) -> None:
    content, is_error = _compute_with(concepts_db, GOOD, OP25, {**OP24, "value": 0.0})
    assert is_error and "previous is zero" in content, content


DEFINE_ARGS: dict[str, object] = {
    "name": "AdvancesToOperatingProfit",
    "scope": "banking",
    "label": "Advances over operating profit",
    "formula": "ratio",
    "bindings": [
        {"variable": "numerator", "concept": "Advances", "at": "current"},
        {"variable": "denominator", "concept": "OperatingProfit", "at": "current"},
    ],
}


def test_a_defined_metric_is_written_unchecked_and_can_be_planned(
    concepts_db: sqlite3.Connection,
) -> None:
    tools = _tools(concepts_db)
    assert _call(tools, DEFINE, DEFINE_ARGS)["tier"] == "derived-concept-ai"
    row = concepts_db.execute(
        "SELECT formula, tier, question, written_by FROM derived_concepts WHERE name = ?",
        ("AdvancesToOperatingProfit",),
    ).fetchone()
    assert row == ("ratio", "derived-concept-ai", QUESTION, WRITER.written_by)
    planned = _call(tools, PLAN, {**ASK, **ONE, "name": "AdvancesToOperatingProfit"})
    assert planned["metric"] == "AdvancesToOperatingProfit" and planned["kind"] == "mixed"
    assert [f["cite_as"] for f in cast(list[dict[str, object]], planned["fetch"])] == [
        ADV25_Q4["cite_as"],
        OP25["cite_as"],
    ], "a balance in a mixed metric is read at the end of the flow's span"
    asked = [f["fundamentals"] for f in cast(list[dict[str, dict[str, object]]], planned["fetch"])]
    assert [(f["concept"], f["quarter"], f["period"]) for f in asked] == [
        ("Advances", "Q4", None),
        ("OperatingProfit", "Q4", "12M"),
    ], "fetched as fundamentals takes them: a balance with no period"


@pytest.mark.parametrize(
    ("changed", "said"),
    [
        pytest.param({"name": "advances ratio"}, "not a CamelCase", id="a name in words"),
        pytest.param({"name": "AdvancesRatioFY25"}, "names a period", id="a period in the name"),
        pytest.param({"name": "OperatingProfitGrowth"}, "already defined", id="a name held"),
        pytest.param({"label": " "}, "label", id="no label"),
        pytest.param({"formula": "mean"}, "no formula", id="a formula not in the library"),
        pytest.param(
            {"bindings": [{"variable": "numerator", "concept": "Advances", "at": "current"}]},
            "takes",
            id="a variable unbound",
        ),
        pytest.param(
            {"bindings": [{"variable": "numerator", "concept": "Advances", "at": "current"}] * 2},
            "bound twice",
            id="a variable bound twice",
        ),
        pytest.param(
            {
                "bindings": [
                    {"variable": "numerator", "concept": "Dividend", "at": "current"},
                    {"variable": "denominator", "concept": "OperatingProfit", "at": "current"},
                ]
            },
            "no concept Dividend",
            id="a concept not held",
        ),
        pytest.param(
            {
                "bindings": [
                    {"variable": "numerator", "concept": "OperatingProfitGrowth", "at": "current"},
                    {"variable": "denominator", "concept": "OperatingProfit", "at": "current"},
                ]
            },
            "compares two periods itself",
            id="an input comparing two periods",
        ),
    ],
)
def test_define_refuses_a_bad_metric_and_writes_nothing(
    concepts_db: sqlite3.Connection, changed: dict[str, object], said: str
) -> None:
    before = concepts_db.execute("SELECT count(*) FROM derived_concepts").fetchone()
    content, is_error = _tools(concepts_db)(DEFINE, {**DEFINE_ARGS, **changed})
    assert is_error and said in content, content
    assert concepts_db.execute("SELECT count(*) FROM derived_concepts").fetchone() == before


def test_a_proposed_formula_is_written_for_a_person(concepts_db: sqlite3.Connection) -> None:
    tools = _tools(concepts_db)
    args: dict[str, object] = {
        "name": "cagr",
        "expression": "(end / start) ** (1 / years) - 1",
        "variables": ["end", "start", "years"],
    }
    first, again = _call(tools, PROPOSE, args), _call(tools, PROPOSE, args)
    assert (first["tier"], first["new"], again["new"]) == ("formula-ai", True, False)
    assert concepts_db.execute("SELECT name, tier, question FROM formula_proposals").fetchall() == [
        ("cagr", "formula-ai", QUESTION)
    ]
    content, is_error = tools(PROPOSE, {**args, "expression": ""})
    assert is_error and "expression" in content


def test_chained_routes_the_metric_tools_and_passes_the_rest_on(
    concepts_db: sqlite3.Connection,
) -> None:
    dispatch = chained(_tools(concepts_db), lambda name, arguments: (f"served {name}", False))
    assert json.loads(dispatch(PLAN, ASK)[0])["plan_id"] == "P1"
    assert dispatch("fundamentals", {}) == ("served fundamentals", False)


def _done(verdict: Verdict | None, *citations: str) -> Analysed:
    claims = [Claim(id="c1", text="Up 15.77%.", citations=list(citations))]
    verdicts = {} if verdict is None else {"c1": Judged(verdict=verdict, reason="r")}
    return Analysed(Memo(summary="s", claims=claims, caveats=[]), verdicts, None, 0, [], None, "d")


def _derived(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    return conn.execute(
        """SELECT ticker, basis, concept, label, period_kind, period_start, period_end, unit,
                  tier, formula, run_id FROM facts WHERE tier = 'derived-ai'"""
    ).fetchall()


def test_a_supported_figure_is_written_under_its_metric_s_name(
    concepts_db: sqlite3.Connection,
) -> None:
    tools = _tools(concepts_db, OP25, OP24)
    _computed(tools)
    written, skipped = write_back(concepts_db, _done("supported", "DERIVED:D1"), tools, WRITER)
    assert skipped == [] and [w["metric"] for w in written] == ["OperatingProfitGrowth"]
    assert written[0]["periods"] == "FY2025 over FY2024"
    assert _derived(concepts_db) == [
        (
            "ICICIBANK",
            "standalone",
            "OperatingProfitGrowth",
            "OperatingProfitGrowth in words, FY2025 over FY2024",
            "duration",
            "2024-04-01",
            "2025-03-31",
            "pure",
            "derived-ai",
            json.dumps(
                {
                    "formula": "growth",
                    "inputs": {"current": OP25["cite_as"], "previous": OP24["cite_as"]},
                    "metric": "OperatingProfitGrowth",
                    "periods": "FY2025 over FY2024",
                    "scope": "banking",
                },
                sort_keys=True,
            ),
            "desk-1",
        )
    ]
    inputs = concepts_db.execute(
        """SELECT f.period_end FROM derived_from d JOIN facts f ON f.id = d.input_fact_id
           ORDER BY f.period_end"""
    ).fetchall()
    assert inputs == [("2024-03-31",), ("2025-03-31",)]


def test_a_ratio_of_balances_is_written_as_at_the_period_s_end(
    concepts_db: sqlite3.Connection,
) -> None:
    tools = _tools(concepts_db, ADV25_Q4, DEP25_Q4)
    planned = _call(tools, PLAN, {**ASK, **SNAP, "name": "CreditDepositRatio"})
    _call(
        tools,
        COMPUTE,
        {
            "plan_id": planned["plan_id"],
            "inputs": [
                {"variable": "numerator", "cite_as": ADV25_Q4["cite_as"]},
                {"variable": "denominator", "cite_as": DEP25_Q4["cite_as"]},
            ],
        },
    )
    write_back(concepts_db, _done("supported", "DERIVED:D1"), tools, WRITER)
    row = _derived(concepts_db)[0]
    assert row[4:7] == ("instant", None, "2025-03-31")


@pytest.mark.parametrize(
    ("verdict", "cites"),
    [
        pytest.param("wrong", ("DERIVED:D1",), id="a claim the critic refused"),
        pytest.param(None, ("DERIVED:D1",), id="a claim never judged"),
        pytest.param("supported", (str(OP25["cite_as"]),), id="a claim citing no computed figure"),
    ],
)
def test_nothing_is_written_without_a_supported_claim_citing_it(
    concepts_db: sqlite3.Connection, verdict: Verdict | None, cites: tuple[str, ...]
) -> None:
    tools = _tools(concepts_db, OP25, OP24)
    _computed(tools)
    assert write_back(concepts_db, _done(verdict, *cites), tools, WRITER) == ([], [])
    assert _derived(concepts_db) == []


def test_a_figure_two_claims_cite_is_written_once(concepts_db: sqlite3.Connection) -> None:
    tools = _tools(concepts_db, OP25, OP24)
    _computed(tools)
    claims = [Claim(id=f"c{n}", text="Up.", citations=["DERIVED:D1"]) for n in (1, 2)]
    judged = {c.id: Judged(verdict="supported", reason="r") for c in claims}
    done = Analysed(Memo(summary="s", claims=claims, caveats=[]), judged, None, 0, [], None, "d")
    assert len(write_back(concepts_db, done, tools, WRITER)[0]) == 1
    assert len(_derived(concepts_db)) == 1


def test_a_figure_already_held_is_not_written_again(concepts_db: sqlite3.Connection) -> None:
    first = _tools(concepts_db, OP25, OP24)
    _computed(first)
    [written], _ = write_back(concepts_db, _done("supported", "DERIVED:D1"), first, WRITER)
    again = _tools(concepts_db, OP25, OP24)
    _computed(again)
    none, skipped = write_back(concepts_db, _done("supported", "DERIVED:D1"), again, WRITER)
    assert none == [] and skipped == [
        {"metric": "OperatingProfitGrowth", "reason": "already held", "fact_id": written["fact_id"]}
    ]
    assert len(_derived(concepts_db)) == 1


def test_a_figure_whose_input_lost_its_check_is_skipped(concepts_db: sqlite3.Connection) -> None:
    tools = _tools(concepts_db, OP25, OP24)
    _computed(tools)
    with concepts_db:
        concepts_db.execute(
            "UPDATE facts SET tier = 'xbrl-ai', verified_by = NULL, verified_at = NULL "
            "WHERE concept = 'OperatingProfit' AND period_end = '2024-03-31'"
        )
    written, skipped = write_back(concepts_db, _done("supported", "DERIVED:D1"), tools, WRITER)
    assert written == [] and skipped == [
        {"metric": "OperatingProfitGrowth", "reason": "an input is no longer verified"}
    ]
    assert _derived(concepts_db) == []


def test_a_memo_never_written_writes_nothing(concepts_db: sqlite3.Connection) -> None:
    tools = _tools(concepts_db, OP25, OP24)
    _computed(tools)
    none = Analysed(None, {}, None, 0, [], "stopped", None)
    assert write_back(concepts_db, none, tools, WRITER) == ([], [])


def test_a_checked_concept_s_figure_is_trusted_and_written_derived(
    concepts_db: sqlite3.Connection,
) -> None:
    with concepts_db:
        concepts_db.execute(
            "UPDATE derived_concepts SET tier = 'derived-concept', verified_by = 'vedant', "
            "verified_at = 't' WHERE name = 'OperatingProfitGrowth'"
        )
    tools = _tools(concepts_db, OP25, OP24)
    assert _computed(tools)["trusted"] is True
    written, _ = write_back(concepts_db, _done("supported", "DERIVED:D1"), tools, WRITER)
    assert written[0]["tier"] == "derived"
    assert concepts_db.execute(
        "SELECT tier, verified_by, verified_at FROM facts WHERE id = ?", (written[0]["fact_id"],)
    ).fetchone() == ("derived", "OperatingProfitGrowth[banking]@vedant", WRITER.written_at)


def _verified_growth(conn: sqlite3.Connection) -> int:
    """OperatingProfitGrowth FY2025 over FY2024, computed, written and its concept checked."""
    tools = _tools(conn, OP25, OP24)
    _computed(tools)
    [written], _ = write_back(conn, _done("supported", "DERIVED:D1"), tools, WRITER)
    verify_derived_concept(conn, "OperatingProfitGrowth", "banking", "Vedant", "t")
    return int(cast(int, written["fact_id"]))


def test_a_verified_figure_already_held_is_fetched_not_planned_again(
    concepts_db: sqlite3.Connection,
) -> None:
    fact_id = _verified_growth(concepts_db)
    tools = _tools(concepts_db)

    planned = _call(tools, PLAN, ASK)

    assert planned["plan_id"] is None and tools.plans == {}, "nothing to compute"
    assert planned["held"] == {"fact_id": fact_id, "tier": "derived"}
    [fetch] = cast(list[dict[str, object]], planned["fetch"])
    assert fetch["fundamentals"] == {
        "company": "ICICIBANK",
        "concept": "OperatingProfitGrowth",
        "year": "2025",
        "quarter": "Q4",
        "period": "12M",
        "basis": "standalone",
        "segment": None,
    }
    db = Path(concepts_db.execute("PRAGMA database_list").fetchone()[2])
    lookup = Lookup(db, concept_index(BagOfWordsEmbedder()), run_id="r", asked_by="test")
    said = json.loads(FundamentalsTool(lookup)(cast(dict[str, object], fetch["fundamentals"])))
    [fact] = said["facts"]
    assert fact["cite_as"] == fetch["cite_as"] and fact["tier"] == "derived", fact
    assert fact["value"] == pytest.approx((672988200000 - 581307600000) / 581307600000)


@pytest.mark.parametrize(
    ("change", "why"),
    [
        pytest.param({"previous_year": "2023"}, "another interval", id="over another period"),
        pytest.param({"basis": "consolidated"}, "another basis", id="on another basis"),
    ],
)
def test_a_plan_unlike_the_held_figure_is_planned_as_ever(
    concepts_db: sqlite3.Connection, change: dict[str, object], why: str
) -> None:
    _verified_growth(concepts_db)
    content, is_error = _tools(concepts_db)(PLAN, {**ASK, **change})
    planned = json.loads(content) if not is_error else {}
    assert not is_error and planned["plan_id"] == "P1" and "held" not in planned, why


def test_a_held_snapshot_is_fetched_as_at_the_quarter_s_end(
    concepts_db: sqlite3.Connection,
) -> None:
    asked = {**ASK, **SNAP, "name": "CreditDepositRatio"}
    tools = _tools(concepts_db, ADV25_Q4, DEP25_Q4)
    planned = _call(tools, PLAN, asked)
    inputs = [
        {"variable": f["variable"], "cite_as": f["cite_as"]}
        for f in cast(list[dict[str, object]], planned["fetch"])
    ]
    _call(tools, COMPUTE, {"plan_id": planned["plan_id"], "inputs": inputs})
    write_back(concepts_db, _done("supported", "DERIVED:D1"), tools, WRITER)
    verify_derived_concept(concepts_db, "CreditDepositRatio", "banking", "Vedant", "t")

    held = _call(_tools(concepts_db), PLAN, asked)

    [fetch] = cast(list[dict[str, dict[str, object]]], held["fetch"])
    assert held["plan_id"] is None and (
        fetch["fundamentals"]["quarter"],
        fetch["fundamentals"]["period"],
    ) == ("Q4", None)


def test_an_unverified_figure_is_computed_again(concepts_db: sqlite3.Connection) -> None:
    tools = _tools(concepts_db, OP25, OP24)
    _computed(tools)
    write_back(concepts_db, _done("supported", "DERIVED:D1"), tools, WRITER)
    planned = _call(_tools(concepts_db), PLAN, ASK)
    assert planned["plan_id"] == "P1" and "held" not in planned


def test_an_unchecked_concept_s_figure_is_not_trusted(concepts_db: sqlite3.Connection) -> None:
    assert _computed(_tools(concepts_db, OP25, OP24))["trusted"] is False


@pytest.mark.parametrize(
    ("changed", "said"),
    [
        pytest.param({"scope": "insurance"}, "scope is", id="an unknown scope"),
        pytest.param({"scope": "*"}, "no concept", id="every scope, one not filing them"),
        pytest.param({"scope": "ind-as"}, "no concept", id="concepts another scope files"),
    ],
)
def test_define_refuses_a_scope_that_does_not_fit(
    concepts_db: sqlite3.Connection, changed: dict[str, object], said: str
) -> None:
    content, is_error = _tools(concepts_db)(DEFINE, {**DEFINE_ARGS, **changed})
    assert is_error and said in content, content


def test_a_metric_built_on_a_metric_is_planned_computed_and_written(
    concepts_db: sqlite3.Connection,
) -> None:
    tools = _tools(concepts_db, OP25, ADV25_Q4, DEP25_Q4)
    _call(
        tools,
        DEFINE,
        {
            "name": "OperatingProfitLessAdvances",
            "scope": "banking",
            "label": "operating profit less advances",
            "formula": "difference",
            "bindings": [
                {"variable": "minuend", "concept": "OperatingProfit", "at": "current"},
                {"variable": "subtrahend", "concept": "Advances", "at": "current"},
            ],
        },
    )
    _call(
        tools,
        DEFINE,
        {
            "name": "ThatOverDeposits",
            "scope": "banking",
            "label": "operating profit less advances over deposits",
            "formula": "ratio",
            "bindings": [
                {
                    "variable": "numerator",
                    "concept": "OperatingProfitLessAdvances",
                    "at": "current",
                },
                {"variable": "denominator", "concept": "Deposits", "at": "current"},
            ],
        },
    )
    planned = _call(tools, PLAN, {**ASK, **ONE, "name": "ThatOverDeposits"})
    fetch = cast(list[dict[str, object]], planned["fetch"])
    assert [(f["variable"], f["cite_as"]) for f in fetch] == [
        ("numerator.minuend", OP25["cite_as"]),
        ("numerator.subtrahend", ADV25_Q4["cite_as"]),
        ("denominator", DEP25_Q4["cite_as"]),
    ]
    shown = _call(
        tools,
        COMPUTE,
        {
            "plan_id": planned["plan_id"],
            "inputs": [{"variable": f["variable"], "cite_as": f["cite_as"]} for f in fetch],
        },
    )
    assert shown["value"] == pytest.approx((672988200000.0 - 800.0) / 1000.0)
    written, _ = write_back(concepts_db, _done("supported", "DERIVED:D1"), tools, WRITER)
    links = concepts_db.execute(
        "SELECT count(*) FROM derived_from WHERE fact_id = ?", (written[0]["fact_id"],)
    ).fetchone()
    assert written[0]["metric"] == "ThatOverDeposits" and links == (3,), "linked to every leaf"


def test_an_every_scope_metric_is_defined_and_planned_for_any_company(
    concepts_db: sqlite3.Connection,
) -> None:
    with concepts_db:
        concepts_db.execute(
            """INSERT INTO facts (ticker, basis, concept, label, period_kind, period_start,
                      period_end, value, unit, tier, written_by, written_at, run_id)
               VALUES ('ITC', 'standalone', 'OperatingProfit', 'OP', 'duration', '2024-04-01',
                       '2025-03-31', 5, 'INR', 'xbrl-ai', 'w', 't', 'r')"""
        )
    tools = _tools(concepts_db)
    args = {
        **DEFINE_ARGS,
        "name": "OperatingProfitOverItself",
        "scope": "ind-as,banking",
        "bindings": [
            {"variable": "numerator", "concept": "OperatingProfit", "at": "current"},
            {"variable": "denominator", "concept": "OperatingProfit", "at": "previous"},
        ],
    }
    assert _call(tools, DEFINE, args)["scope"] == "*"
    planned = _call(tools, PLAN, {**ASK, "name": "OperatingProfitOverItself", "company": "ITC"})
    assert planned["metric"] == "OperatingProfitOverItself" and planned["scope"] == "ind-as"
