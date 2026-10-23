"""The `fundamentals` tool over the facts database: its answers, its refusals and its schema."""

from __future__ import annotations

import json
from typing import cast, get_args

import pytest

from aigent.analyst.fundamentals import (
    FUNDAMENTALS_TOOL,
    Company,
    FinancialYear,
    FundamentalsTool,
)
from aigent.facts.cfg_facts import FACTS_TICKERS, FACTS_YEARS
from aigent.facts.lookup import BASES, PERIODS, Lookup

ASKED: dict[str, object] = {
    "company": "ITC",
    "year": "2025",
    "what": "revenue from operations",
    "period": "year",
    "basis": "consolidated",
    "segment": None,
}


def test_an_answer_is_the_facts_found_and_each_is_recorded(facts_lookup: Lookup) -> None:
    tool = FundamentalsTool(facts_lookup)

    said = cast(dict[str, list[dict[str, object]]], json.loads(tool(ASKED)))

    assert said["facts"][0]["value"] == 816.0
    assert tool.looked_up == [str(f["cite_as"]) for f in said["facts"]]
    assert tool.looked_up[0] == "FACT:ITC:2025:consolidated:year:RevenueFromOperations"


def test_a_segment_and_a_basis_reach_the_lookup(facts_lookup: Lookup) -> None:
    tool = FundamentalsTool(facts_lookup)

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
