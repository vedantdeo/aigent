"""Project 2's ADK agent on the fake client: ADK drives the loop, `llm` still makes every call."""

from __future__ import annotations

from typing import cast

import pytest
from anthropic.types import (
    Message,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    WebSearchToolResultBlock,
)

pytest.importorskip("google.adk")

from aigent.agent import FINISH, SYSTEM  # noqa: E402
from aigent.agent_adk import run  # noqa: E402
from aigent.config import MODEL  # noqa: E402

from .conftest import (  # noqa: E402
    FAKE_USAGE,
    FakeSearch,
    MakeLlm,
    Sent,
    tool_turn,
    turns,
)

SEARCH_THEN_SUM = tool_turn(
    ("s1", "search_reports", {"query": "dividend", "report": "RELIANCE-FY25"}),
    ("c1", "calculate", {"expression": "5.5 * 2"}),
)


def test_a_turns_tools_run_and_the_answer_comes_back(make_llm: MakeLlm, search: FakeSearch) -> None:
    llm, fake = make_llm(turns(SEARCH_THEN_SUM, "Rs 11 [RELIANCE-FY25#0001]"))

    traced = run(llm, search, "twice the dividend?")

    assert traced.answer == "Rs 11 [RELIANCE-FY25#0001]" and traced.stopped is None
    assert traced.tools == ["search_reports", "calculate"]
    assert [call.step for call in llm.trace] == ["agent:1", "agent:2"], "every call through llm"
    results = cast(list[dict[str, object]], fake.sent[1].messages[-1]["content"])
    assert [r["tool_use_id"] for r in results] == ["s1", "c1"], "both results in one message"


def test_a_web_search_is_recorded_and_its_blocks_never_reach_adk(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    """ADK's converter refuses server-side search blocks; the turn keeps them, ADK does not."""
    searched = Message(
        id="msg_web",
        type="message",
        role="assistant",
        model=MODEL,
        content=[
            ServerToolUseBlock(
                type="server_tool_use", id="w1", name="web_search", input={"query": "rate"}
            ),
            WebSearchToolResultBlock(type="web_search_tool_result", tool_use_id="w1", content=[]),
            TextBlock(type="text", text="about 58 rupees", citations=None),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=FAKE_USAGE,
    )
    llm, _ = make_llm(lambda sent: searched)

    traced = run(llm, search, "the AUD rate?")

    assert traced.tools == ["web_search"]
    assert traced.answer == "about 58 rupees"


def _until_told(sent: Sent) -> Message | str:
    told = FINISH in str(sent.messages[-1]["content"])
    return "all I found" if told else tool_turn(("t1", "current_time", {}))


def test_the_last_turn_under_the_cap_is_an_answer(make_llm: MakeLlm, search: FakeSearch) -> None:
    llm, _ = make_llm(_until_told)

    traced = run(llm, search, "keep going", max_turns=3)

    assert traced.answer == "all I found"
    assert "last of 3 turns" in str(traced.stopped)
    assert [call.step for call in llm.trace][-1] == "agent:3 answer"


def test_every_turn_is_sent_as_the_other_agents_send_it(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, fake = make_llm(turns(SEARCH_THEN_SUM, "done"))

    run(llm, search, "q")

    for sent in fake.sent:
        assert sent.system == SYSTEM
        tools = cast(list[dict[str, object]], sent.tools)
        assert [t["name"] for t in tools] == [
            "search_reports",
            "calculate",
            "current_time",
            "read_file",
            "web_search",
        ]
        assert sent.thinking == {"type": "adaptive"}
        assert sent.cache_control == {"type": "ephemeral"}


def test_a_turn_that_searched_goes_back_to_the_api_exactly_as_it_came(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    """The live run's 400: ADK held the turn without its search blocks, and the API refuses a latest
    turn whose thinking has moved. The next request carries the turn as it was sent."""
    searched_then_asks = Message(
        id="msg_1",
        type="message",
        role="assistant",
        model=MODEL,
        content=[
            ThinkingBlock(type="thinking", thinking="need the rate", signature="sig"),
            ServerToolUseBlock(
                type="server_tool_use", id="w1", name="web_search", input={"query": "rate"}
            ),
            WebSearchToolResultBlock(type="web_search_tool_result", tool_use_id="w1", content=[]),
            TextBlock(type="text", text="about 58", citations=None),
            ToolUseBlock(
                type="tool_use", id="c1", name="calculate", input={"expression": "2.86*58"}
            ),
        ],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=FAKE_USAGE,
    )
    llm, fake = make_llm(turns(searched_then_asks, "Rs 16.6 crore"))

    traced = run(llm, search, "A$2.86m in rupees?")

    assert traced.answer == "Rs 16.6 crore"
    assert traced.tools == ["web_search", "calculate"]
    sent_back = cast(list[dict[str, object]], fake.sent[1].messages[1]["content"])
    kinds = [block["type"] for block in sent_back]
    assert kinds == ["thinking", "server_tool_use", "web_search_tool_result", "text", "tool_use"]
