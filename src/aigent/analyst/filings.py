"""The analyst's search: all six filings, FY25 and FY24 for each company, scoped by company and by
the financial year a report ends, its passages fenced as text a third party wrote."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Protocol, cast

from anthropic.types import ToolParam

from aigent.config import SEARCH_METHOD
from aigent.retrieval.chunk import Chunk, Inventory, by_sentence, context_block
from aigent.retrieval.corpus import MANIFEST, load_corpus
from aigent.retrieval.rank import Indexes, build_ranker
from aigent.tools import fence

_DOCUMENTS = cast(
    dict[str, dict[str, object]], json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
)
FILINGS: tuple[str, ...] = tuple(sorted(_DOCUMENTS))
COMPANIES: dict[str, str] = {
    str(entry["ticker"]): str(entry["company"]) for entry in _DOCUMENTS.values()
}
_YEAR = re.compile(r"-FY(\d\d)$")
ALL = "all"


def year_of(doc_id: str) -> int:
    """The financial year a report ends: ITC-FY25 is 2025."""
    found = _YEAR.search(doc_id)
    if found is None:
        raise ValueError(f"{doc_id} names no financial year")
    return 2000 + int(found.group(1))


YEARS: tuple[int, ...] = tuple(sorted({year_of(doc_id) for doc_id in FILINGS}))


def scope(company: str, year: str) -> frozenset[str] | None:
    """The reports a search reads: one company or all, one year or all; None for all six."""
    if company != ALL and company not in COMPANIES:
        raise ValueError(f"no company {company!r}; one of {', '.join(COMPANIES)} or {ALL}")
    if year != ALL and (not year.isdigit() or int(year) not in YEARS):
        raise ValueError(f"no filings for year {year!r}; one of {', '.join(map(str, YEARS))}")
    if company == ALL and year == ALL:
        return None
    return frozenset(
        doc_id
        for doc_id in FILINGS
        if (company == ALL or _DOCUMENTS[doc_id]["ticker"] == company)
        and (year == ALL or year_of(doc_id) == int(year))
    )


class Search(Protocol):
    """A query in, passages out, from the reports `company` and `year` pick."""

    def __call__(self, query: str, /, *, company: str = ALL, year: str = ALL) -> list[Chunk]: ...


def build_filings_search(cache: bool) -> Search:
    """Chunk, embed and index the six filings once; passages come back by `SEARCH_METHOD`."""
    inventory = Inventory.build(
        "by_sentence", load_corpus(cache=cache, doc_ids=FILINGS), by_sentence()
    )
    ranker = build_ranker(SEARCH_METHOD, Indexes.build(inventory, cache=cache))
    one_at_a_time = threading.Lock()  # torch on MPS is not safe across threads

    def search(query: str, /, *, company: str = ALL, year: str = ALL) -> list[Chunk]:
        reports = scope(company, year)
        with one_at_a_time:
            hits = ranker.rank(query, doc_id=reports)
        return [inventory.by_id[hit.chunk_id] for hit in hits]

    return search


SEARCH_FILINGS_TOOL: ToolParam = {
    "name": "search_filings",
    "description": (
        "Search the FY25 and FY24 annual reports for passages that answer a query. Returns up to "
        "five passages, each tagged with the id to cite it by. Scope by company and by the "
        "financial year a report ends, so a figure comes from the year asked about."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in a report's words."},
            "company": {
                "type": "string",
                "enum": [*COMPANIES, ALL],
                "description": "One company's ticker, or all.",
            },
            "year": {
                "type": "string",
                "enum": [*map(str, YEARS), ALL],
                "description": "The financial year a report ends (2025 is FY 2024-25), or all.",
            },
        },
        "required": ["query", "company", "year"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass(frozen=True)
class Searched:
    """One `search_filings` call: what was asked, of which reports, and the passage ids found."""

    query: str
    company: str
    year: str
    found: list[str]


@dataclass
class FilingSearch:
    """`search_filings` over one `Search`, recording every search; a bad scope comes back as an
    error message, never an exception."""

    search: Search
    searches: list[Searched] = field(default_factory=list[Searched])

    def __call__(self, arguments: dict[str, object]) -> str:
        query = str(arguments.get("query", ""))
        company, year = str(arguments.get("company", ALL)), str(arguments.get("year", ALL))
        try:
            passages = self.search(query, company=company, year=year)
        except ValueError as error:
            return f"Error: {error}"
        self.searches.append(Searched(query, company, year, [chunk.id for chunk in passages]))
        found = context_block(passages) or "No passages matched that query."
        return fence("search_filings", found)
