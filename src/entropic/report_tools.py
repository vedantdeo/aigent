"""Tools over the annual reports, for a model to call: their schemas, and what runs them.

Unlike `tools`, these need the rest of the package — a ranker and the reports — so they live apart,
and `tools` stays free to lift into another framework unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anthropic.types import ToolParam

from entropic.retrieval.chunk import context_block
from entropic.workflows.reports import REPORTS, Search

SEARCH_TOOL: ToolParam = {
    "name": "search_reports",
    "description": (
        "Search the annual reports for passages that answer a query. Returns up to five passages, "
        "each tagged with the id to cite it by. Different words find different passages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in a report's words."},
            "report": {
                "type": "string",
                "enum": [*REPORTS, "all"],
                "description": "One report's id to search only it, or all.",
            },
        },
        "required": ["query", "report"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass(frozen=True)
class Searched:
    """One `search_reports` call: what was asked, of which report, and the passage ids it found."""

    query: str
    report: str
    found: list[str]


@dataclass
class ReportSearch:
    """`search_reports` over one `Search`, keeping a record of every search it runs."""

    search: Search
    searches: list[Searched] = field(default_factory=list[Searched])

    def __call__(self, arguments: dict[str, object]) -> str:
        query, report = str(arguments.get("query", "")), str(arguments.get("report", "all"))
        passages = self.search(query, doc_id=None if report == "all" else report)
        self.searches.append(Searched(query, report, [chunk.id for chunk in passages]))
        return context_block(passages) or "No passages matched that query."
