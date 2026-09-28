"""Project 2's LangGraph agent on the fake client: the loop rules `run_tools` keeps, as nodes.

Each test scripts a different conversation, so they are not a table.
"""

from __future__ import annotations

from typing import cast

import pytest
from anthropic.types import Message, ServerToolUseBlock, TextBlock

pytest.importorskip("langgraph")

from aigent.agent import FINISH, SYSTEM, TOOLS  # noqa: E402
from aigent.agent_graph import run  # noqa: E402
from aigent.config import MODEL  # noqa: E402

from .conftest import (  # noqa: E402
    FAKE_USAGE,
    FakeSearch,
    MakeLlm,
    Sent,
    tool_results,
    tool_turn,
    turns,
)

SEARCH_THEN_SUM = tool_turn(
    ("s1", "search_reports", {"query": "dividend", "report": "RELIANCE-FY25"}),
    ("c1", "calculate", {"expression": "5.5 * 2"}),
)


def test_a_turns_tools_all_run_and_answer_in_one_message(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, fake = make_llm(turns(SEARCH_THEN_SUM, "Rs 11 [RELIANCE-FY25#0001]"))

    traced = run(llm, search, "twice the dividend?")

    assert traced.answer == "Rs 11 [RELIANCE-FY25#0001]" and traced.stopped is None
    assert traced.tools == ["search_reports", "calculate"], "the trajectory, in order"
    ids = [result["tool_use_id"] for result in tool_results(fake.sent[1])]
    assert ids == ["s1", "c1"], "both results in one user message"
    assert [call.step for call in llm.trace] == ["agent:1", "agent:2"]
    assert llm.budget.held_usd == 0.0


def test_a_failed_tool_goes_back_as_an_error_result(make_llm: MakeLlm, search: FakeSearch) -> None:
    llm, fake = make_llm(turns(tool_turn(("c1", "calculate", {"expression": "1 / 0"})), "sorry"))

    run(llm, search, "divide by zero")

    [result] = tool_results(fake.sent[1])
    assert result["is_error"] is True and str(result["content"]).startswith("Error:")


def _until_told(sent: Sent) -> Message | str:
    told = FINISH in str(sent.messages[-1]["content"])
    return "all I found" if told else tool_turn(("t1", "current_time", {}))


def test_the_last_turn_under_the_cap_is_an_answer(make_llm: MakeLlm, search: FakeSearch) -> None:
    llm, fake = make_llm(_until_told)

    traced = run(llm, search, "keep searching", max_turns=3)

    assert traced.answer == "all I found"
    assert "last of 3 turns" in str(traced.stopped)
    told = cast(list[object], fake.sent[-1].messages[-1]["content"])
    assert told[-1] == {"type": "text", "text": FINISH}
    assert traced.tools == ["current_time", "current_time"], "the answer turn called nothing"
    assert [call.step for call in llm.trace] == ["agent:1", "agent:2", "agent:3 answer"]


def test_a_first_turn_the_budget_cannot_afford_is_never_sent(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, fake = make_llm(lambda sent: "unused", limit_usd=0.0001)

    traced = run(llm, search, "anything")

    assert traced.answer is None and "ceiling" in str(traced.stopped)
    assert not fake.sent


def test_a_server_tool_the_api_ran_is_in_the_trajectory(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    searched = Message(
        id="msg_web",
        type="message",
        role="assistant",
        model=MODEL,
        content=[
            ServerToolUseBlock(
                type="server_tool_use", id="w1", name="web_search", input={"query": "price"}
            ),
            TextBlock(type="text", text="up 3% today", citations=None),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=FAKE_USAGE,
    )
    llm, _ = make_llm(lambda sent: searched)

    traced = run(llm, search, "the share price now?")

    assert traced.tools == ["web_search"] and traced.answer is not None


def test_every_turn_is_sent_as_the_runner_agent_sends_it(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, fake = make_llm(turns(SEARCH_THEN_SUM, "done"))

    run(llm, search, "q")

    for sent in fake.sent:
        assert sent.system == SYSTEM
        assert [t["name"] for t in cast(list[dict[str, object]], sent.tools)][-1] == TOOLS[-1][
            "name"
        ]
        assert sent.thinking == {"type": "adaptive"}
        assert sent.cache_control == {"type": "ephemeral"}
