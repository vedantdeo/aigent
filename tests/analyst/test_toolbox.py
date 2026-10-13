"""The generalist's toolbox: a server's tools beside the two specialists', evidence kept from each
search, and a specialist call answered through the router."""

from __future__ import annotations

import json
from typing import cast

import pytest

from aigent.analyst.specialists import FigureJob, Specialist, Told
from aigent.analyst.toolbox import (
    EXTRACT_FIGURES_TOOL,
    READ_HEADLINE_TOOL,
    Passage,
    passages_in,
)
from aigent.errors import SpecialistUnreachable
from aigent.mcp_client import connect
from aigent.mcp_server import build_server
from aigent.retrieval.chunk import context_block

from ..conftest import PASSAGES, FakeFilings, FakeSearch
from .conftest import DIVIDEND, FIGURE, box, record, search, told


def test_a_search_result_s_passages_are_read_back_with_their_ids() -> None:
    chunks = [c for chunks in PASSAGES.values() for c in chunks]

    found = passages_in(
        f'<untrusted source="search_filings">\n{context_block(chunks)}\n</untrusted>'
    )

    assert [(p.id, p.text) for p in found] == [(c.id, c.text) for c in chunks]
    assert found[1].report == "RELIANCE-FY25" and found[1].source == chunks[1].citation
    assert passages_in("No passages matched that query.") == []


def test_the_generalist_is_shown_the_server_s_tools_and_both_specialists() -> None:
    names = [tool["name"] for tool in box(search).tools]

    assert names == ["search_filings", "read_headline", "extract_figures"]
    assert dict(READ_HEADLINE_TOOL["input_schema"])["required"] == ["headline"]
    assert dict(EXTRACT_FIGURES_TOOL["input_schema"])["required"] == ["chunk_ids", "wanted"]


def test_figures_are_extracted_from_passages_a_search_returned_each_tagged_with_its_own() -> None:
    asked: list[FigureJob] = []

    def figures(job: FigureJob) -> Told:
        asked.append(job)
        return told('{"figures": [' + (FIGURE if "dividend" in job.passage else "") + "]}")

    tools = box(search, figures)
    tools("search_filings", {"query": "dividend", "company": "all", "year": "all"})

    content, is_error = tools(
        "extract_figures", {"chunk_ids": [DIVIDEND, "ITC-FY25#0001"], "wanted": "the dividend"}
    )

    assert not is_error, content
    assert json.loads(content) == {"figures": [{**json.loads(FIGURE), "chunk_id": DIVIDEND}]}
    assert [(job.source, job.wanted) for job in asked] == [
        ("RELIANCE-FY25", "the dividend"),
        ("ITC-FY25", "the dividend"),
    ]
    assert asked[0].passage == PASSAGES["RELIANCE-FY25"][0].text
    assert [r.specialist for r in tools.route.routed] == ["extract_figures"] * 2


def test_a_headline_is_read_through_the_router() -> None:
    tools = box(search)

    content, is_error = tools("read_headline", {"headline": "  Infy Q2 profit rises 11%\n"})

    assert not is_error and json.loads(content) == {
        "ticker": "INFY",
        "headline": "Infy Q2 profit rises 11%",
    }
    assert [r.path for r in tools.route.routed] == ["specialist"]


@pytest.mark.parametrize(
    ("name", "arguments", "says"),
    [
        pytest.param("read_headline", {"headline": "  "}, "one line of text", id="no headline"),
        pytest.param(
            "extract_figures", {"chunk_ids": [], "wanted": "x"}, "at least one", id="no passages"
        ),
        pytest.param(
            "extract_figures", {"chunk_ids": [DIVIDEND], "wanted": ""}, "what", id="nothing wanted"
        ),
        pytest.param(
            "extract_figures",
            {"chunk_ids": [DIVIDEND, "INFY-FY25#0009"], "wanted": "x"},
            "no passage INFY-FY25#0009",
            id="a passage no search returned",
        ),
    ],
)
def test_a_bad_specialist_call_is_an_error_result_and_reaches_no_model(
    name: str, arguments: dict[str, object], says: str
) -> None:
    tools = box(search)
    tools("search_filings", {"query": "dividend"})

    content, is_error = tools(name, arguments)

    assert is_error and content.startswith("Error: ") and says in content, content
    assert tools.route.routed == []


def test_an_answer_neither_model_gets_past_its_check_is_an_error_result() -> None:
    def generalist(job: str) -> Told:
        return told("")

    tools = box(search)
    tools.read_headline = Specialist("read_headline", lambda job: told(""), generalist, record)

    content, is_error = tools("read_headline", {"headline": "Infy Q2 profit rises 11%"})

    assert is_error and "no headline record" in content
    assert tools.route.routed[0].path is None


def test_an_unreachable_specialist_fails_the_task_rather_than_the_call() -> None:
    def down(job: str) -> Told:
        raise SpecialistUnreachable("hostllm-headline did not answer in 2 tries")

    with pytest.raises(SpecialistUnreachable):
        box(search, headline=down)("read_headline", {"headline": "Infy Q2 profit rises 11%"})


def test_a_failed_search_adds_no_evidence_and_other_tools_pass_straight_through() -> None:
    def served(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        if name == "calculate":
            return "42.0", False
        return "Error: no company 'INFY'", True

    tools = box(served)

    assert tools("search_filings", {"query": "q", "company": "INFY"})[1] is True
    assert tools("calculate", {"expression": "6 * 7"}) == ("42.0", False)
    assert tools.evidence == {}


def test_over_the_real_server_a_search_feeds_the_figures_call() -> None:
    with connect(build_server(FakeSearch(), FakeFilings()), fenced=False) as served:
        tools = box(served.dispatch)
        tools.served_tools = served.tools
        found, failed = tools("search_filings", {"query": "dividend", "company": "RELIANCE"})
        content, is_error = tools(
            "extract_figures", {"chunk_ids": [DIVIDEND], "wanted": "dividend"}
        )

    assert not failed and list(tools.evidence) == [DIVIDEND], found
    assert not is_error and cast(dict[str, list[object]], json.loads(content))["figures"], content
    assert isinstance(tools.evidence[DIVIDEND], Passage)


def test_a_fundamentals_answer_is_kept_as_evidence_under_its_cite_as() -> None:
    answer = '{"cite_as": "ITC:2025:dividend", "value": 14.35, "source": "ITC-FY25 p51"}'
    replies = iter([(answer, False), ("14.35", False), ("Error: no metric", True), ("{}", False)])
    tools = box(lambda name, arguments: next(replies))

    for _ in range(4):
        tools("fundamentals", {"ticker": "ITC", "metric": "dividend", "year": "2025"})

    assert list(tools.evidence) == ["ITC:2025:dividend"]
    kept = tools.evidence["ITC:2025:dividend"]
    assert (kept.source, kept.text) == ("ITC-FY25 p51", answer)
