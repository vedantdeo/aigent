"""`fundamentals` and `compare_unverified` over the facts database: verified answers, held-back
unverified ones, the comparison, the refusals and the schemas."""

from __future__ import annotations

import json
from typing import cast, get_args

import pytest

from aigent.analyst.cfg_analyst import ANALYST_SERVED_TOOLS
from aigent.analyst.fundamentals import (
    COMPARE_TOOL,
    FUNDAMENTALS_TOOL,
    Company,
    FinancialYear,
    FundamentalsTool,
    agrees,
)
from aigent.facts.cfg_facts import FACTS_TICKERS, FACTS_YEARS
from aigent.facts.lookup import BASES, PERIODS, Lookup
from aigent.facts.store import CHECKED_TIERS

ASKED: dict[str, object] = {
    "company": "ITC",
    "year": "2025",
    "what": "revenue from operations",
    "period": "year",
    "basis": "consolidated",
    "segment": None,
}


def test_a_verified_answer_is_shown_and_recorded(checked_facts_lookup: Lookup) -> None:
    tool = FundamentalsTool(checked_facts_lookup)

    said = cast(dict[str, list[dict[str, object]]], json.loads(tool(ASKED)))

    assert said["facts"][0]["value"] == 816.0
    assert said["facts"][0]["tier"] == "xbrl-pdf"
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

    segment = json.loads(tool(ASKED | {"what": "segment revenue", "segment": "agri"}))
    standalone = json.loads(tool(ASKED | {"basis": "standalone"}))

    assert segment["facts"][0]["segment"] == "Agri Business"
    assert standalone["facts"][0]["value"] == 700.0


@pytest.mark.parametrize(
    ("change", "says"),
    [
        pytest.param({"company": "INFY"}, "no company 'INFY'", id="a company not held"),
        pytest.param({"year": "2019"}, "no year '2019'", id="a year not held"),
        pytest.param({"year": "FY25"}, "no year 'FY25'", id="a year not a number"),
        pytest.param({"what": "  "}, "say which figure", id="no figure named"),
        pytest.param({"period": "Q5"}, "no period 'Q5'", id="a period that is not one"),
        pytest.param({"basis": "group"}, "no basis 'group'", id="a basis that is not one"),
        pytest.param({"segment": 3}, "a name or null", id="a segment that is not a name"),
        pytest.param({"company": "WIPRO"}, "Closest labels filed", id="nothing held for it"),
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
    assert properties["period"]["enum"] == list(PERIODS)
    assert properties["basis"]["enum"] == list(BASES)

    assert set(schema["required"]) == set(properties)


def test_compare_asks_for_the_agents_own_figure_and_its_source() -> None:
    schema = cast(dict[str, object], COMPARE_TOOL["input_schema"])
    assert schema["required"] == ["lookup_id", "my_value", "how"]
    assert "Never cite an unverified entry" in str(COMPARE_TOOL.get("description"))


def test_the_analyst_can_compare_whenever_it_can_look_up() -> None:
    assert {"fundamentals", "compare_unverified"} <= set(ANALYST_SERVED_TOOLS)
