"""The capstone's search over six filings, scoped by company and year, its passages fenced."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest

from aigent.analyst import filings
from aigent.analyst.filings import (
    FILINGS,
    SEARCH_FILINGS_TOOL,
    FilingSearch,
    build_filings_search,
    scope,
    year_of,
)
from aigent.retrieval.chunk import Chunk
from aigent.retrieval.hits import Hit
from aigent.retrieval.rank import Scope

CHUNKS = {
    "ITC-FY24#0001": Chunk("ITC-FY24#0001", "Dividend of 13.75 a share.", "ITC-FY24", 1, 0),
    "ITC-FY25#0001": Chunk("ITC-FY25#0001", "Dividend of 14.35 a share.", "ITC-FY25", 1, 0),
}


def test_the_filings_are_both_years_of_all_three_companies() -> None:
    assert FILINGS == (
        "ITC-FY24",
        "ITC-FY25",
        "RELIANCE-FY24",
        "RELIANCE-FY25",
        "TATAMOTORS-FY24",
        "TATAMOTORS-FY25",
    )


@pytest.mark.parametrize(
    ("doc_id", "year"), [("ITC-FY25", 2025), ("RELIANCE-FY24", 2024), ("TATAMOTORS-FY24", 2024)]
)
def test_a_report_s_year_is_the_financial_year_it_ends(doc_id: str, year: int) -> None:
    assert year_of(doc_id) == year


@pytest.mark.parametrize(
    ("company", "year", "expected"),
    [
        pytest.param("all", "all", None, id="everything"),
        pytest.param("ITC", "all", {"ITC-FY24", "ITC-FY25"}, id="one company, both years"),
        pytest.param(
            "all", "2025", {"ITC-FY25", "RELIANCE-FY25", "TATAMOTORS-FY25"}, id="one year"
        ),
        pytest.param("RELIANCE", "2024", {"RELIANCE-FY24"}, id="one report"),
    ],
)
def test_scope(company: str, year: str, expected: set[str] | None) -> None:
    assert scope(company, year) == (None if expected is None else frozenset(expected))


@pytest.mark.parametrize(
    ("company", "year"),
    [
        pytest.param("INFY", "all", id="an unknown company"),
        pytest.param("ITC", "2023", id="a missing year"),
    ],
)
def test_an_unknown_scope_is_refused(company: str, year: str) -> None:
    with pytest.raises(ValueError, match=company if company == "INFY" else year):
        scope(company, year)


@dataclass
class _Ranker:
    asked: list[Scope] = field(default_factory=list[Scope])
    loaded: list[object] = field(default_factory=list[object])

    def rank(self, query: str, doc_id: Scope = None) -> list[Hit]:
        self.asked.append(doc_id)
        wanted = None if doc_id is None else {doc_id} if isinstance(doc_id, str) else set(doc_id)
        return [Hit(i, 1.0) for i, c in CHUNKS.items() if wanted is None or c.doc_id in wanted]


@pytest.fixture
def ranker(monkeypatch: pytest.MonkeyPatch) -> _Ranker:
    built = _Ranker()

    class _Inventory:
        by_id = CHUNKS

    monkeypatch.setattr(
        filings, "load_corpus", lambda cache, doc_ids: built.loaded.append(doc_ids) or []
    )
    monkeypatch.setattr(filings.Inventory, "build", lambda *args: _Inventory())
    monkeypatch.setattr(filings.Indexes, "build", lambda *args, **kwargs: None)
    monkeypatch.setattr(filings, "build_ranker", lambda method, indexes: built)
    return built


def test_the_search_reads_all_six_filings_and_ranks_within_the_scope(ranker: _Ranker) -> None:
    search = build_filings_search(cache=False)

    found = search("dividend", company="ITC", year="2025")

    assert [c.id for c in found] == ["ITC-FY25#0001"]
    assert ranker.asked == [frozenset({"ITC-FY25"})]
    assert ranker.loaded == [FILINGS]


def test_the_tool_fences_the_passages_and_records_each_search(ranker: _Ranker) -> None:
    tool = FilingSearch(build_filings_search(cache=False))

    said = tool({"query": "dividend", "company": "ITC", "year": "all"})

    assert said.startswith('<untrusted source="search_filings">') and "ITC-FY24#0001" in said
    assert [(s.query, s.company, s.year, s.found) for s in tool.searches] == [
        ("dividend", "ITC", "all", ["ITC-FY24#0001", "ITC-FY25#0001"])
    ]


def test_the_tool_answers_a_bad_scope_with_an_error_not_an_exception(ranker: _Ranker) -> None:
    tool = FilingSearch(build_filings_search(cache=False))

    said = tool({"query": "dividend", "company": "INFY", "year": "all"})

    assert said.startswith("Error:") and "INFY" in said and tool.searches == []


def test_the_schema_offers_exactly_the_companies_and_years_there_are() -> None:
    schema = cast(dict[str, dict[str, dict[str, list[str]]]], SEARCH_FILINGS_TOOL["input_schema"])
    properties = schema["properties"]
    assert properties["company"]["enum"] == ["ITC", "RELIANCE", "TATAMOTORS", "all"]
    assert properties["year"]["enum"] == ["2024", "2025", "all"]
