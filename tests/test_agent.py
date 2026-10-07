"""The first agent, on the SDK's real tool runner over the fake client: the model picks searches.

What is tested is the part that is ours — that a search the model asks for reaches it, what a
stopped run still shows, and how the request is sent. The search tool itself is tested in
`test_report_tools`; whether the model searches well is a question for a live run.
"""

from __future__ import annotations

from typing import cast

import pytest
from anthropic.types import (
    CitationsWebSearchResultLocation,
    ContentBlock,
    Message,
    ServerToolUsage,
    ServerToolUseBlock,
    TextBlock,
    TextCitation,
    ToolUseBlock,
    Usage,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
)

from aigent.agent import FINISH, run
from aigent.config import MAX_AGENT_TURNS, MAX_TOKENS_TOOL_LOOP, MAX_USD_PER_TURN, MODEL
from aigent.pricing import PRICES, worst_case_usd
from aigent.retrieval.chunk import Chunk
from aigent.tools import WEB_SEARCH_TOOL
from aigent.tools_config import MAX_WEB_SEARCHES

from .conftest import FAKE_USAGE, FakeSearch, MakeLlm, Sent, tool_results, tool_turn, turns

ANSWER = "Tata Motors, on its small commercial vehicles [TATAMOTORS-FY25#0001]"
EVERY_REPORT = ["ITC-FY25#0001", "RELIANCE-FY25#0001", "TATAMOTORS-FY25#0001"]
SEARCHING = tool_turn(("s1", "search_reports", {"query": "rural demand", "report": "all"}))


def until_told(sent: Sent) -> Message | str:
    return ANSWER if FINISH in str(sent.messages[-1]["content"]) else SEARCHING


# Room for one turn's worst case and not a second, once the first has been billed.
ONE_TURN_USD = worst_case_usd(
    MODEL, FAKE_USAGE.input_tokens, MAX_TOKENS_TOOL_LOOP, cached=True, web_searches=MAX_WEB_SEARCHES
)


def test_a_search_the_model_asks_for_reaches_it_with_ids_to_cite(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, fake = make_llm(turns(SEARCHING, ANSWER))

    result = run(llm, search, "who is most exposed to rural demand?")

    assert [searched.found for searched in result.searches] == [EVERY_REPORT]
    assert (result.turns, result.answer) == (2, ANSWER)
    [passages] = tool_results(fake.sent[1])
    assert 'id="TATAMOTORS-FY25#0001"' in str(passages["content"]), "passages carry ids to cite"


def test_the_other_tools_still_run_beside_search(make_llm: MakeLlm, search: FakeSearch) -> None:
    asked = tool_turn(("c1", "calculate", {"expression": "138913 / 159043"}))
    llm, fake = make_llm(turns(asked, "down about 12.7%"))

    result = run(llm, search, "by how much did SCV volumes fall?")

    [ratio] = tool_results(fake.sent[1])
    assert str(ratio["content"]).startswith("0.873"), ratio
    assert search.asked == [] and result.searches == []


@pytest.mark.parametrize(
    ("limit_usd", "says", "turns", "searches"),
    [
        pytest.param(
            1.0,
            f"last of {MAX_AGENT_TURNS} turns",
            MAX_AGENT_TURNS,
            MAX_AGENT_TURNS - 1,
            id="the turn cap, whose last turn is kept for an answer",
        ),
        pytest.param(
            ONE_TURN_USD + 0.001, "per-run ceiling", 2, 1, id="a search the budget cannot afford"
        ),
    ],
)
def test_a_run_out_of_room_answers_from_what_it_searched(
    make_llm: MakeLlm, search: FakeSearch, limit_usd: float, says: str, turns: int, searches: int
) -> None:
    llm, _ = make_llm(until_told, limit_usd=limit_usd)

    result = run(llm, search, "who is most exposed to rural demand?")

    assert result.answer == ANSWER and says in str(result.stopped), result.stopped
    assert (result.turns, len(result.searches)) == (turns, searches)


def test_with_no_room_even_to_answer_it_still_shows_what_it_searched(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, fake = make_llm(until_told)

    def growing(query: str, /, *, doc_id: str | None = None) -> list[Chunk]:
        fake.input_tokens = int(MAX_USD_PER_TURN / (PRICES[MODEL].cache_write * 1e-6)) + 1_000
        return search(query, doc_id=doc_id)

    result = run(llm, growing, "who is most exposed to rural demand?")

    assert result.answer is None and "per-turn ceiling" in str(result.stopped), result.stopped
    assert (result.turns, [searched.found for searched in result.searches]) == (1, [EVERY_REPORT])


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        pytest.param(
            "thinking",
            {"type": "adaptive"},
            id="thinking stays on, or Opus 5 can write a tool call as text",
        ),
        pytest.param(
            "cache_control",
            {"type": "ephemeral"},
            id="the conversation is cached, so no turn pays twice for the last",
        ),
    ],
)
def test_how_the_agent_sends_its_turns(
    make_llm: MakeLlm, search: FakeSearch, field: str, expected: object
) -> None:
    llm, fake = make_llm(lambda sent: "an answer")

    run(llm, search, "anything")

    assert getattr(fake.sent[0], field) == expected


PAGE_A, PAGE_B = "https://a.example/ril", "https://b.example/ril"


def _cited(text: str, *urls: str) -> TextBlock:
    citations: list[TextCitation] = [
        CitationsWebSearchResultLocation(
            type="web_search_result_location",
            url=url,
            title="a page",
            encrypted_index="x",
            cited_text=text,
        )
        for url in urls
    ]
    return TextBlock(type="text", text=text, citations=citations)


def _searched(*urls: str) -> list[ContentBlock]:
    """A web search and the pages it returned, as the API puts them in a turn."""
    results = [
        WebSearchResultBlock(
            type="web_search_result", url=url, title="a page", encrypted_content="x"
        )
        for url in urls
    ]
    return [
        ServerToolUseBlock(
            type="server_tool_use", id="srv1", name="web_search", input={"q": "ril"}
        ),
        WebSearchToolResultBlock(
            type="web_search_tool_result", tool_use_id="srv1", content=results
        ),
    ]


def _first_turn(blocks: list[ContentBlock], searches: int) -> Message:
    """A turn that searched the web, then asks for the reports too, so the answer comes later."""
    report = ToolUseBlock(
        type="tool_use", id="s1", name="search_reports", input={"query": "q", "report": "all"}
    )
    usage = Usage(
        input_tokens=100,
        output_tokens=50,
        server_tool_use=ServerToolUsage(web_search_requests=searches, web_fetch_requests=0),
    )
    return Message(
        id="m",
        type="message",
        role="assistant",
        model=MODEL,
        content=[*blocks, report],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=usage,
    )


@pytest.mark.parametrize(
    ("first", "cited", "returned", "searches"),
    [
        pytest.param(
            _first_turn([*_searched(PAGE_A, PAGE_B), _cited("Up 4%", PAGE_A)], 1),
            [PAGE_A],
            [PAGE_B],
            1,
            id="a page cited in the turn that searched, answered a turn later",
        ),
        pytest.param(
            _first_turn(_searched(PAGE_A), 1),
            [],
            [PAGE_A],
            1,
            id="a page returned and never cited",
        ),
        pytest.param(
            _first_turn([], 2),
            [],
            [],
            2,
            id="searches whose results were left out of the response",
        ),
    ],
)
def test_a_web_answer_shows_what_its_searches_found_across_every_turn(
    make_llm: MakeLlm,
    search: FakeSearch,
    first: Message,
    cited: list[str],
    returned: list[str],
    searches: int,
) -> None:
    llm, fake = make_llm(turns(first, "About -4% since the results."))

    result = run(llm, search, "how has Reliance's share price moved?")

    assert (result.web.cited, result.web.returned) == (cited, returned)
    assert result.answer == "About -4% since the results."
    assert result.web.searches == searches, result.web
    assert WEB_SEARCH_TOOL in cast(list[object], fake.sent[0].tools), "web search is offered"
