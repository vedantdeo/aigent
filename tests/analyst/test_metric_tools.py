"""`analyst.metric_tools`: a plan's fetch list, a computation checked against its plan, a metric
defined and a formula proposed, every refusal as an error result, and the write-back of what a
supported claim cites."""

from __future__ import annotations

import json
import sqlite3
from typing import Literal, cast

import pytest

pytest.importorskip("langgraph")

from aigent.analyst.graph import Analysed  # noqa: E402
from aigent.analyst.memo import Claim, Judged, Memo  # noqa: E402
from aigent.analyst.metric_tools import (  # noqa: E402
    COMPUTE,
    DEFINE,
    PLAN,
    PROPOSE,
    MetricTools,
    chained,
    write_back,
)
from aigent.facts.store import Writer  # noqa: E402

from .conftest import ADV25, DEP25, EPS25, OP24, OP25, UNVERIFIED, evidence_of  # noqa: E402

WRITER = Writer("analyst-desk|m", "2026-11-03T22:00:00+05:30", "desk-1")
QUESTION = "How much did ICICI Bank's operating profit grow in FY25?"
ASK: dict[str, object] = {
    "what": "operating profit growth",
    "company": "ICICIBANK",
    "year": "2025",
    "period": "year",
    "basis": "standalone",
}
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
    assert planned["fetch"] == [
        {
            "variable": "current",
            "fundamentals": {
                "company": "ICICIBANK",
                "year": "2025",
                "what": "OperatingProfit",
                "period": "year",
                "basis": "standalone",
                "segment": None,
            },
            "cite_as": OP25["cite_as"],
        },
        {
            "variable": "previous",
            "fundamentals": {
                "company": "ICICIBANK",
                "year": "2024",
                "what": "OperatingProfit",
                "period": "year",
                "basis": "standalone",
                "segment": None,
            },
            "cite_as": OP24["cite_as"],
        },
    ]


def test_no_metric_fitting_points_to_the_library(concepts_db: sqlite3.Connection) -> None:
    planned = _call(_tools(concepts_db), PLAN, {**ASK, "what": "dividend payout"})
    assert planned["metric"] is None and "growth(current, previous)" in str(planned["library"])


@pytest.mark.parametrize(
    ("changed", "said"),
    [
        pytest.param({"what": " "}, "give what", id="no words"),
        pytest.param({"company": "ACME"}, "give what", id="a company not held"),
        pytest.param({"year": "FY25"}, "give what", id="a year not in digits"),
        pytest.param({"period": "Q5"}, "period is one of", id="an unknown period"),
        pytest.param({"basis": "group"}, "period is one of", id="an unknown basis"),
        pytest.param(
            {"what": "OperatingProfitGrowth", "period": "H1"},
            None,
            id="a year-on-year metric plans for a half too",
        ),
    ],
)
def test_plan_refuses_a_bad_ask(
    concepts_db: sqlite3.Connection, changed: dict[str, object], said: str | None
) -> None:
    content, is_error = _tools(concepts_db)(PLAN, {**ASK, **changed})
    assert is_error is (said is not None) and (said is None or said in content), content


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
    tools = _tools(concepts_db, ADV25, DEP25)
    planned = _call(tools, PLAN, {**ASK, "what": "CreditDepositRatio"})
    shown = _call(
        tools,
        COMPUTE,
        {
            "plan_id": planned["plan_id"],
            "inputs": [
                {"variable": "numerator", "cite_as": ADV25["cite_as"]},
                {"variable": "denominator", "cite_as": DEP25["cite_as"]},
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
    planned = _call(tools, PLAN, {**ASK, "what": "OperatingProfitLessEps"})
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
    "label": "Advances over operating profit",
    "formula": "ratio",
    "bindings": [
        {"variable": "numerator", "concept": "Advances", "offset": "t"},
        {"variable": "denominator", "concept": "OperatingProfit", "offset": "t"},
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
    planned = _call(tools, PLAN, {**ASK, "what": "AdvancesToOperatingProfit"})
    assert planned["metric"] == "AdvancesToOperatingProfit"


@pytest.mark.parametrize(
    ("changed", "said"),
    [
        pytest.param({"name": "advances ratio"}, "not a CamelCase", id="a name in words"),
        pytest.param({"name": "AdvancesRatioFY25"}, "names a period", id="a period in the name"),
        pytest.param({"name": "OperatingProfitGrowth"}, "already defined", id="a name held"),
        pytest.param({"label": " "}, "label", id="no label"),
        pytest.param({"formula": "mean"}, "no formula", id="a formula not in the library"),
        pytest.param(
            {"bindings": [{"variable": "numerator", "concept": "Advances", "offset": "t"}]},
            "takes",
            id="a variable unbound",
        ),
        pytest.param(
            {"bindings": [{"variable": "numerator", "concept": "Advances", "offset": "t"}] * 2},
            "bound twice",
            id="a variable bound twice",
        ),
        pytest.param(
            {
                "bindings": [
                    {"variable": "numerator", "concept": "Dividend", "offset": "t"},
                    {"variable": "denominator", "concept": "OperatingProfit", "offset": "t"},
                ]
            },
            "no concept Dividend",
            id="a concept not held",
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
    assert concepts_db.execute("SELECT name, tier, question FROM formulas").fetchall() == [
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
    assert _derived(concepts_db) == [
        (
            "ICICIBANK",
            "standalone",
            "OperatingProfitGrowth",
            "OperatingProfitGrowth in words",
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
    tools = _tools(concepts_db, ADV25, DEP25)
    planned = _call(tools, PLAN, {**ASK, "what": "CreditDepositRatio"})
    _call(
        tools,
        COMPUTE,
        {
            "plan_id": planned["plan_id"],
            "inputs": [
                {"variable": "numerator", "cite_as": ADV25["cite_as"]},
                {"variable": "denominator", "cite_as": DEP25["cite_as"]},
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
