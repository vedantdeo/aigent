"""The generalist's toolbox and the list of experts in it: a server's tools beside the
specialists', evidence kept from each search and from what a specialist's answer cites, and a
specialist call answered through the router."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import pytest

from aigent.analyst import experts as listed
from aigent.analyst.experts import (
    BUILDERS,
    EXTRACT_FIGURES_TOOL,
    experts,
    figures_expert,
)
from aigent.analyst.served import report_analyst
from aigent.analyst.specialists import FigureJob, Specialist, Told, check_figures
from aigent.analyst.toolbox import Passage, Toolbox, passages_in
from aigent.errors import SpecialistUnreachable
from aigent.llm.core import Dispatch, Llm
from aigent.mcp_client import connect
from aigent.mcp_server import build_server
from aigent.retrieval.chunk import context_block

from ..conftest import (
    PASSAGES,
    FakeFilings,
    FakeOpenAI,
    FakeSearch,
    MakeLlm,
    asking,
    completion,
)
from .conftest import DIVIDEND, FIGURE, box, never, search, told


def test_a_search_result_s_passages_are_read_back_with_their_ids() -> None:
    chunks = [c for chunks in PASSAGES.values() for c in chunks]

    found = passages_in(
        f'<untrusted source="search_filings">\n{context_block(chunks)}\n</untrusted>'
    )

    assert [(p.id, p.text) for p in found] == [(c.id, c.text) for c in chunks]
    assert found[1].report == "RELIANCE-FY25" and found[1].source == chunks[1].citation
    assert passages_in("No passages matched that query.") == []


def test_the_generalist_is_shown_the_server_s_tools_and_the_specialist() -> None:
    names = [tool["name"] for tool in box(search).tools]

    assert names == ["search_filings", "extract_figures"]
    said = EXTRACT_FIGURES_TOOL.get("description", "")
    assert said.startswith("Specialist for ") and "Use it" in said and "Limits:" in said, said
    assert dict(EXTRACT_FIGURES_TOOL["input_schema"])["required"] == ["question", "ticker", "year"]


ASK: dict[str, object] = {
    "question": "  What dividend did Reliance recommend for FY25?\n",
    "ticker": "RELIANCE",
}
CALL: dict[str, object] = {**ASK, "year": "2025"}
WRONG = FIGURE.replace("5.5", "9.9")  # a dividend no passage prints


def test_a_call_is_one_job_whose_figures_come_back_with_the_passages_they_cite_as_evidence() -> (
    None
):
    asked: list[FigureJob] = []

    def figures(job: FigureJob) -> Told:
        asked.append(job)
        return told('{"figures": [' + FIGURE + "]}")

    tools = box(search, figures)

    content, is_error = tools("extract_figures", CALL)

    assert not is_error, content
    assert asked == [FigureJob("What dividend did Reliance recommend for FY25?", "RELIANCE", 2025)]
    assert json.loads(content) == {"figures": [{**json.loads(FIGURE), "basis": None}]}
    assert list(tools.evidence) == [DIVIDEND], "a claim can cite the passage it was read from"
    assert tools.evidence[DIVIDEND].text == PASSAGES["RELIANCE-FY25"][0].text
    assert [r.specialist for r in tools.route.routed] == ["extract_figures"]


def _failing() -> Toolbox:
    """A toolbox whose figures specialist and fallback both answer a dividend nothing prints."""
    tools = box(search)
    reads = lambda job: told('{"figures": [' + WRONG + "]}")  # noqa: E731
    tools.experts = [figures_expert(Specialist("extract_figures", reads, reads, check_figures))]
    return tools


def test_an_answer_neither_model_gets_past_its_check_is_an_error_and_adds_no_evidence() -> None:
    tools = _failing()

    content, is_error = tools("extract_figures", CALL)

    assert is_error and f"{DIVIDEND} does not back 9.9" in content, content
    assert tools.route.routed[0].path is None and tools.evidence == {}


def test_an_exact_repeat_of_a_failed_call_is_refused_unrun() -> None:
    tools = _failing()

    first, _ = tools("extract_figures", CALL)
    ran = len(tools.route.routed)
    again, is_error = tools("extract_figures", dict(reversed(CALL.items())))
    other, _ = tools("extract_figures", {**CALL, "year": "2024"})

    assert (
        is_error
        and again.startswith("Error: this exact call already failed (")
        and (first.removeprefix("Error: ") in again)
    ), again
    assert len(tools.route.routed) == ran + 1, "the repeat ran nothing; the changed call ran"
    assert not other.startswith("Error: this exact call")


def test_a_call_that_succeeded_may_be_made_again() -> None:
    tools = box(search)

    for _ in range(2):
        _, is_error = tools("extract_figures", CALL)
        assert not is_error
    assert len(tools.route.routed) == 2


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param({**CALL, "question": "  "}, "as you were asked", id="no question"),
        pytest.param({"question": "q", "year": "2025"}, "ticker must be", id="no ticker"),
        pytest.param({**CALL, "ticker": "INFY"}, "ticker must be one of", id="a company not filed"),
        pytest.param({**CALL, "year": "2023"}, "year must be one of", id="a year not filed"),
        pytest.param({**CALL, "year": "FY25"}, "year must be one of", id="a year not a number"),
        pytest.param(ASK, "year must be one of", id="no year"),
    ],
)
def test_a_bad_specialist_call_is_an_error_result_and_reaches_no_model(
    arguments: dict[str, object], says: str
) -> None:
    tools = box(search)

    content, is_error = tools("extract_figures", arguments)

    assert is_error and content.startswith("Error: ") and says in content, content
    assert tools.route.routed == []


def test_an_unreachable_specialist_fails_the_task_rather_than_the_call() -> None:
    def down(job: FigureJob) -> Told:
        raise SpecialistUnreachable("hostllm-analyst did not answer in 2 tries")

    with pytest.raises(SpecialistUnreachable):
        box(search, down)("extract_figures", CALL)


def test_a_failed_search_adds_no_evidence_and_other_tools_pass_straight_through() -> None:
    def served(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        if name == "calculate":
            return "42.0", False
        return "Error: no company 'INFY'", True

    tools = box(served)

    assert tools("search_filings", {"query": "q", "company": "INFY"})[1] is True
    assert tools("calculate", {"expression": "6 * 7"}) == ("42.0", False)
    assert tools.evidence == {}


def test_over_the_real_server_the_specialist_s_own_search_finds_what_it_cites(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    searching = asking(("search_report", {"query": "dividend"}))
    fake = FakeOpenAI(searching, completion('{"figures": [' + FIGURE + "]}"))
    small = Llm("hostllm-analyst", sdk=cast(object, fake))
    filings = FakeFilings()
    with connect(build_server(FakeSearch(), filings), fenced=False) as served:
        tools = box(served.dispatch)
        tools.served_tools = served.tools
        tools.experts = [figures_expert(report_analyst(small, make_llm(never)[0], served.dispatch))]
        content, is_error = tools("extract_figures", CALL)

    assert not is_error and json.loads(content)["figures"], content
    assert filings.asked == [("dividend", "RELIANCE", "2025")]
    assert isinstance(tools.evidence[DIVIDEND], Passage) and list(tools.evidence) == [DIVIDEND]


def test_a_fundamentals_answer_is_kept_as_evidence_under_its_cite_as() -> None:
    answer = '{"cite_as": "ITC:2025:dividend", "value": 14.35, "source": "ITC-FY25 p51"}'
    replies = iter([(answer, False), ("14.35", False), ("Error: no metric", True), ("{}", False)])
    tools = box(lambda name, arguments: next(replies))

    for _ in range(4):
        tools("fundamentals", {"ticker": "ITC", "metric": "dividend", "year": "2025"})

    assert list(tools.evidence) == ["ITC:2025:dividend"]
    kept = tools.evidence["ITC:2025:dividend"]
    assert (kept.source, kept.text) == ("ITC-FY25 p51", answer)


@pytest.mark.parametrize(
    "served",
    [
        pytest.param(
            ["search_filings", "extract_figures"], id="a server's tool named as an expert is"
        ),
        pytest.param(["search_filings", "search_filings"], id="a server's tool listed twice"),
    ],
)
def test_two_tools_with_one_name_are_refused(served: list[str]) -> None:
    tools = box(search)

    with pytest.raises(ValueError, match="two tools are called"):
        Toolbox(
            [{"name": name, "input_schema": {"type": "object"}} for name in served],
            search,
            tools.route,
            tools.experts,
        )


def test_the_list_builds_every_specialist_on_its_own_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[str, str, str, Dispatch]] = []
    for name in ("report_analyst",):

        def build(
            small: str, big: str, served: Dispatch, model: str | None = "unset", name: str = name
        ) -> Specialist[object, object]:
            built.append((name, small, f"{big} as {model}", served))
            return Specialist(name, never, never, lambda job, said: said)

        monkeypatch.setattr(listed, name, build)

    small = cast("Callable[[str], Llm]", lambda client: client)
    made = experts(cast(Llm, "generalist"), search, small, model="claude-haiku-4-5")

    assert list(BUILDERS) == sorted(BUILDERS), "the list keeps one canonical order"
    assert built == [
        ("report_analyst", "hostllm-analyst", "generalist as claude-haiku-4-5", search),
    ]
    assert [expert.name for expert in made] == ["extract_figures"]
