"""The frozen fundamentals table and its tool: every value cited to a page of a filing."""

from __future__ import annotations

import json
import re
from typing import cast

import pytest

from aigent.analyst.fundamentals import (
    FUNDAMENTALS,
    FUNDAMENTALS_TOOL,
    Fundamental,
    FundamentalsTool,
    load_fundamentals,
    lookup,
)
from aigent.retrieval.corpus import CORPUS_DIR, load_pdf_cached

TABLE = load_fundamentals()
FIELDS = ["value", "unit", "printed", "doc_id", "page", "note"]


def test_the_table_holds_four_metrics_for_two_years_of_three_companies() -> None:
    assert len(TABLE) == 24
    assert {(f.ticker, f.year) for f in TABLE.values()} == {
        (t, y) for t in ("ITC", "RELIANCE", "TATAMOTORS") for y in (2024, 2025)
    }


def test_the_file_is_sorted_at_every_level_with_one_shape_per_entry() -> None:
    raw = cast(
        dict[str, dict[str, dict[str, dict[str, object]]]],
        json.loads(FUNDAMENTALS.read_text())["fundamentals"],
    )
    assert list(raw) == sorted(raw)
    for years in raw.values():
        assert list(years) == sorted(years)
        for metrics in years.values():
            assert list(metrics) == sorted(metrics)
            assert all(list(entry) == FIELDS for entry in metrics.values())


def test_each_value_is_cited_to_its_own_year_s_report() -> None:
    for f in TABLE.values():
        assert f.doc_id == f"{f.ticker}-FY{f.year % 100}", f.key


def test_a_lookup_returns_the_cited_value() -> None:
    found = lookup(TABLE, "ITC", "dividend", 2025)

    assert (found.value, found.unit, found.doc_id, found.page) == (14.35, "INR", "ITC-FY25", 51)
    assert found.key == "ITC:2025:dividend" and found.citation == "ITC-FY25 p51"


@pytest.mark.parametrize(
    ("ticker", "metric", "year", "named"),
    [
        pytest.param("INFY", "eps", 2025, "no company 'INFY'", id="an unknown company"),
        pytest.param(
            "ITC", "ebitda", 2025, "no metric 'ebitda'", id="a metric the table does not hold"
        ),
        pytest.param("ITC", "eps", 2023, "no 2023 figures for ITC", id="a year it does not hold"),
    ],
)
def test_a_lookup_outside_the_table_is_refused(
    ticker: str, metric: str, year: int, named: str
) -> None:
    with pytest.raises(ValueError, match=named):
        lookup(TABLE, ticker, metric, year)


def test_the_tool_answers_in_json_with_its_citation_and_records_each_lookup() -> None:
    tool = FundamentalsTool(TABLE)

    said = json.loads(tool({"ticker": "RELIANCE", "metric": "eps", "year": "2024"}))

    assert said["value"] == 102.90 and said["cite_as"] == "RELIANCE:2024:eps"
    assert said["source"] == "RELIANCE-FY24 p111"
    assert tool.looked_up == ["RELIANCE:2024:eps"]


def test_the_tool_answers_an_unknown_lookup_with_an_error_not_an_exception() -> None:
    tool = FundamentalsTool(TABLE)

    said = tool({"ticker": "RELIANCE", "metric": "ebitda", "year": "2024"})

    assert said.startswith("Error:") and "ebitda" in said and tool.looked_up == []


def test_the_schema_offers_exactly_what_the_table_holds() -> None:
    schema = cast(dict[str, dict[str, dict[str, list[str]]]], FUNDAMENTALS_TOOL["input_schema"])
    properties = schema["properties"]
    assert properties["ticker"]["enum"] == ["ITC", "RELIANCE", "TATAMOTORS"]
    assert properties["metric"]["enum"] == ["dividend", "eps", "profit", "revenue"]
    assert properties["year"]["enum"] == ["2024", "2025"]


def _page(f: Fundamental) -> str:
    document = load_pdf_cached(CORPUS_DIR / f"{f.doc_id}.pdf")
    starts = (*document.page_starts, len(document.text))
    return document.text[starts[f.page - 1] : starts[f.page]].replace(",", "")


@pytest.mark.skipif(
    not all((CORPUS_DIR / f"{f.doc_id}.pdf").exists() for f in TABLE.values()),
    reason="the filings are not fetched (./scripts/fetch-corpus.sh)",
)
@pytest.mark.parametrize("fundamental", list(TABLE.values()), ids=lambda f: f.key)
def test_every_printed_number_is_on_the_cited_page(fundamental: Fundamental) -> None:
    page = _page(fundamental)
    numbers = [n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", fundamental.printed)]
    missing = [n for n in numbers if n not in page and n.rstrip("0").rstrip(".") not in page]
    assert not missing, f"{fundamental.key}: {missing} not on {fundamental.citation}"
