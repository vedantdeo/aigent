"""The report tools, run directly: what a `search_reports` call searches, and what it hands back."""

from __future__ import annotations

import pytest

from entropic.report_tools import ReportSearch, Searched
from entropic.retrieval.chunk import Chunk

from .conftest import FakeSearch

EVERY_REPORT = ["ITC-FY25#0001", "RELIANCE-FY25#0001", "TATAMOTORS-FY25#0001"]


@pytest.mark.parametrize(
    ("report", "scope", "found"),
    [
        pytest.param(
            "TATAMOTORS-FY25",
            "TATAMOTORS-FY25",
            ["TATAMOTORS-FY25#0001"],
            id="a named report is searched alone",
        ),
        pytest.param("all", None, EVERY_REPORT, id="all searches every report"),
    ],
)
def test_a_search_runs_where_the_model_asked(
    search: FakeSearch, report: str, scope: str | None, found: list[str]
) -> None:
    reports = ReportSearch(search)

    handed_back = reports({"query": "rural demand", "report": report})

    assert search.asked == [("rural demand", scope)]
    assert reports.searches == [Searched("rural demand", report, found)]
    missing = [chunk_id for chunk_id in found if f'id="{chunk_id}"' not in handed_back]
    assert missing == [], "every passage carries the id to cite it by"


def test_a_search_that_finds_nothing_says_so() -> None:
    def empty(query: str, /, *, doc_id: str | None = None) -> list[Chunk]:
        return []

    reports = ReportSearch(empty)

    assert reports({"query": "lunar mining", "report": "all"}) == "No passages matched that query."
    assert reports.searches == [Searched("lunar mining", "all", [])]
