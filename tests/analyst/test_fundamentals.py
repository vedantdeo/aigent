"""`get_metric`, `fundamentals` and `compare_unverified` over the facts database: candidate names,
verified answers by exact concept and time, held-back unverified ones, the comparison, the refusals
and the schemas."""

from __future__ import annotations

import json
from typing import cast, get_args

import pytest

from aigent.analyst.cfg_analyst import ANALYST_SERVED_TOOLS, DESK_SERVED_TOOLS
from aigent.analyst.fundamentals import (
    COMPARE_TOOL,
    FUNDAMENTALS_TOOL,
    GET_METRIC_TOOL,
    Company,
    FinancialYear,
    FundamentalsTool,
    agrees,
)
from aigent.facts.cfg_facts import FACTS_TICKERS, FACTS_YEARS
from aigent.facts.lookup import BASES, LENGTHS, QUARTERS, Lookup
from aigent.facts.store import CHECKED_TIERS, Writer, connect, write_derived_concept

ASKED: dict[str, object] = {
    "company": "ITC",
    "concept": "RevenueFromOperations",
    "year": "2025",
    "quarter": "Q4",
    "period": "12M",
    "basis": "consolidated",
    "segment": None,
}


def test_a_verified_answer_is_shown_and_recorded(checked_facts_lookup: Lookup) -> None:
    tool = FundamentalsTool(checked_facts_lookup)

    said = cast(dict[str, list[dict[str, object]]], json.loads(tool(ASKED)))

    assert said["facts"][0]["value"] == 816.0
    assert said["facts"][0]["tier"] == "xbrl-filed"
    assert "note" not in said
    assert tool.looked_up == [str(f["cite_as"]) for f in said["facts"]]
    assert tool.looked_up[0] == "FACT:ITC:2025:consolidated:year:RevenueFromOperations"


def test_unverified_facts_are_held_back_and_never_recorded(facts_lookup: Lookup) -> None:
    tool = FundamentalsTool(facts_lookup)

    said = cast(dict[str, object], json.loads(tool(ASKED)))

    assert said["facts"] == []
    assert said["lookup_id"] == "L1"
    assert cast(int, said["unverified_held"]) >= 1
    assert "compare_unverified" in str(said["note"])
    assert "816" not in json.dumps(said), said
    assert tool.looked_up == []
    assert not {f.tier for f in tool.held["L1"]} & set(CHECKED_TIERS), tool.held["L1"]


def test_each_lookup_holds_its_own_entries(facts_lookup: Lookup) -> None:
    tool = FundamentalsTool(facts_lookup)

    first = json.loads(tool(ASKED))
    second = json.loads(tool(ASKED | {"basis": "standalone"}))

    assert (first["lookup_id"], second["lookup_id"]) == ("L1", "L2")
    assert tool.held["L1"][0].value == 816.0
    assert tool.held["L2"][0].value == 700.0


@pytest.mark.parametrize(
    ("mine", "agreed"),
    [
        pytest.param(816.0, True, id="the same figure"),
        pytest.param(816.0 * 1.004, True, id="within half a percent"),
        pytest.param(816.0 * 1.01, False, id="a percent off"),
        pytest.param(81.6, False, id="a scale out"),
    ],
)
def test_compare_marks_each_held_entry_without_a_citation(
    facts_lookup: Lookup, mine: float, agreed: bool
) -> None:
    tool = FundamentalsTool(facts_lookup)
    tool(ASKED)

    said = json.loads(tool.compare({"lookup_id": "L1", "my_value": mine, "how": "results p.2"}))

    [entry, *_] = said["unverified"]
    assert entry["value"] == 816.0 and entry["tier"] == "xbrl-ai", entry
    assert entry["agrees"] is agreed, entry
    assert all("cite_as" not in e for e in said["unverified"]), said
    assert "never cite" in said["note"]
    assert tool.looked_up == []


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param({"lookup_id": "L9"}, "no lookup 'L9'", id="a lookup never made"),
        pytest.param({"my_value": None}, "as a number", id="no figure of its own"),
        pytest.param({"my_value": "816"}, "as a number", id="a figure as text"),
        pytest.param({"my_value": True}, "as a number", id="a boolean for a figure"),
        pytest.param({"how": "  "}, "where your figure came from", id="no account of it"),
    ],
)
def test_compare_refuses_until_the_agent_has_its_own_figure(
    facts_lookup: Lookup, arguments: dict[str, object], says: str
) -> None:
    tool = FundamentalsTool(facts_lookup)
    tool(ASKED)

    said = tool.compare({"lookup_id": "L1", "my_value": 816.0, "how": "p.2"} | arguments)

    assert said.startswith("Error:") and says in said, said


@pytest.mark.parametrize(
    ("mine", "held", "agreed"),
    [
        (100.0, 100.4, True),
        (100.0, 100.6, False),
        pytest.param(100.0, 100.502, True, id="half a percent of the larger figure"),
        (-50.0, -50.2, True),
        (0.0, 0.0, True),
        (0.0, 1.0, False),
        (50.0, -50.0, False),
    ],
)
def test_agrees(mine: float, held: float, agreed: bool) -> None:
    assert agrees(mine, held) is agreed


def test_a_segment_and_a_basis_reach_the_lookup(checked_facts_lookup: Lookup) -> None:
    tool = FundamentalsTool(checked_facts_lookup)

    segment = json.loads(tool(ASKED | {"concept": "SegmentRevenue", "segment": "agri"}))
    standalone = json.loads(tool(ASKED | {"basis": "standalone"}))

    assert segment["facts"][0]["segment"] == "Agri Business"
    assert standalone["facts"][0]["value"] == 700.0


@pytest.mark.parametrize(
    ("change", "value", "cite"),
    [
        pytest.param({}, 816.0, "2025:consolidated:year:RevenueFromOperations", id="a year"),
        pytest.param(
            {"period": "3M"}, 204.0, "2025:consolidated:Q4:RevenueFromOperations", id="Q4"
        ),
        pytest.param(
            {"quarter": "Q3", "period": "9M"},
            612.0,
            "2025:consolidated:9M:RevenueFromOperations",
            id="nine months ending Q3",
        ),
        pytest.param(
            {"concept": "Assets", "period": None},
            900.0,
            "2025:consolidated:Q4:Assets",
            id="a snapshot as at Q4's end",
        ),
        pytest.param(
            {"concept": "Assets", "quarter": "Q3", "period": None},
            880.0,
            "2025:consolidated:Q3:Assets",
            id="a snapshot as at Q3's end",
        ),
    ],
)
def test_a_concept_is_fetched_at_the_time_and_period_named(
    checked_facts_lookup: Lookup, change: dict[str, object], value: float, cite: str
) -> None:
    said = json.loads(FundamentalsTool(checked_facts_lookup)(ASKED | change))
    assert [(f["value"], f["cite_as"]) for f in said["facts"]] == [(value, f"FACT:ITC:{cite}")]


@pytest.mark.parametrize(
    ("change", "says"),
    [
        pytest.param({"company": "INFY"}, "no company 'INFY'", id="a company not held"),
        pytest.param({"year": "2019"}, "no year '2019'", id="a year not held"),
        pytest.param({"year": "FY25"}, "no year 'FY25'", id="a year not a number"),
        pytest.param({"concept": "  "}, "name the concept", id="no concept named"),
        pytest.param({"quarter": "Q5"}, "no quarter 'Q5'", id="a quarter that is not one"),
        pytest.param({"period": "4M"}, "no period '4M'", id="a period that is not one"),
        pytest.param({"basis": "group"}, "no basis 'group'", id="a basis that is not one"),
        pytest.param({"segment": 3}, "a name or null", id="a segment that is not a name"),
        pytest.param(
            {"concept": "revenue from operations"}, "find its name with get_metric", id="words"
        ),
        pytest.param({"company": "WIPRO"}, "WIPRO files no concept", id="a concept not filed"),
        pytest.param({"concept": "Assets"}, "a snapshot", id="a snapshot given a period"),
        pytest.param({"period": None}, "reads a span", id="a flow given no period"),
        pytest.param({"quarter": "Q3", "period": "6M"}, "ends at Q2 only", id="6M ending Q3"),
        pytest.param({"year": "2024"}, "files no RevenueFromOperations", id="nothing then"),
    ],
)
def test_a_bad_call_is_an_error_message_and_records_nothing(
    facts_lookup: Lookup, change: dict[str, object], says: str
) -> None:
    tool = FundamentalsTool(facts_lookup)

    said = tool(ASKED | change)

    assert said.startswith("Error:") and says in said, said
    assert tool.looked_up == []


def test_the_schema_offers_exactly_what_the_database_holds() -> None:
    schema = cast(dict[str, dict[str, dict[str, object]]], FUNDAMENTALS_TOOL["input_schema"])
    properties = schema["properties"]
    assert properties["company"]["enum"] == list(FACTS_TICKERS) == list(get_args(Company))
    assert properties["year"]["enum"] == [str(y) for y in FACTS_YEARS]
    assert get_args(FinancialYear) == tuple(str(y) for y in FACTS_YEARS)
    assert properties["quarter"]["enum"] == list(QUARTERS)
    assert properties["period"]["anyOf"] == [
        {"type": "string", "enum": list(LENGTHS)},
        {"type": "null"},
    ]
    assert properties["basis"]["enum"] == list(BASES)
    assert set(schema["required"]) == set(properties)
    found = cast(dict[str, dict[str, object]], GET_METRIC_TOOL["input_schema"])
    assert found["required"] == ["company", "what"] and set(found["properties"]) == {
        "company",
        "what",
    }


def test_get_metric_offers_filed_concepts_with_their_kind(facts_lookup: Lookup) -> None:
    tool = FundamentalsTool(facts_lookup)

    revenue = json.loads(tool.candidates({"company": "ITC", "what": "revenue from operations"}))
    assets = json.loads(tool.candidates({"company": "ITC", "what": "total assets"}))

    assert revenue["filed"][0] == {
        "concept": "RevenueFromOperations",
        "label": "Revenue from operations",
        "kind": "flow",
    }
    assert assets["filed"][0]["concept"] == "Assets" and assets["filed"][0]["kind"] == "snapshot"
    assert revenue["scope"] == "ind-as" and revenue["derived"] == []
    none = json.loads(tool.candidates({"company": "WIPRO", "what": "revenue from operations"}))
    assert none["filed"] == [], "only what the company itself files"
    assert tool.looked_up == [], "finding a name fetches nothing"


def test_get_metric_offers_a_derived_metric_of_the_company_s_scope(
    facts_db: object, facts_lookup: Lookup
) -> None:
    conn = connect(facts_lookup.db)
    bindings = (
        '{"current": {"at": "current", "concept": "RevenueFromOperations"}, '
        '"previous": {"at": "previous", "concept": "RevenueFromOperations"}}'
    )
    with conn:
        for scope in ("ind-as", "banking"):
            write_derived_concept(
                conn,
                "RevenueGrowth",
                scope,
                "Growth in revenue",
                "growth",
                bindings,
                None,
                Writer("w", "t", "r"),
            )
    conn.close()

    said = json.loads(
        FundamentalsTool(facts_lookup).candidates({"company": "ITC", "what": "revenue growth"})
    )

    assert said["derived"] == [
        {"name": "RevenueGrowth", "label": "Growth in revenue", "kind": "flow"}
    ]


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param({"company": "INFY", "what": "revenue"}, "no company", id="a company not held"),
        pytest.param({"company": "ITC", "what": " "}, "say which figure", id="no words"),
    ],
)
def test_get_metric_refuses_a_bad_call(
    facts_lookup: Lookup, arguments: dict[str, object], says: str
) -> None:
    said = FundamentalsTool(facts_lookup).candidates(arguments)
    assert said.startswith("Error:") and says in said, said


def test_compare_asks_for_the_agents_own_figure_and_its_source() -> None:
    schema = cast(dict[str, object], COMPARE_TOOL["input_schema"])
    assert schema["required"] == ["lookup_id", "my_value", "how"]
    assert "Never cite an unverified entry" in str(COMPARE_TOOL.get("description"))


def test_the_analyst_can_compare_whenever_it_can_look_up() -> None:
    assert {"fundamentals", "compare_unverified"} <= set(ANALYST_SERVED_TOOLS)


def test_whoever_can_fetch_a_concept_can_find_its_name() -> None:
    assert {"fundamentals", "get_metric"} <= set(ANALYST_SERVED_TOOLS)
    assert {"fundamentals", "get_metric"} <= set(DESK_SERVED_TOOLS)
