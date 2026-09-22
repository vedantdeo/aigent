"""The first agent, on the SDK's real tool runner over the fake client: the model picks searches.

What is tested is the part that is ours — which report a search runs on, what the model is handed
back, what a stopped run still shows, and how the request is sent. Whether the model searches well
is a question for a live run.
"""

from __future__ import annotations

import pytest

from entropic.agent import run
from entropic.config import MAX_AGENT_TURNS, MAX_TOKENS_TOOL_LOOP, MODEL
from entropic.pricing import worst_case_usd

from .conftest import FAKE_USAGE, FakeSearch, MakeLlm, tool_results, tool_turn, turns

ANSWER = "Tata Motors, on its small commercial vehicles [TATAMOTORS-FY25#0001]"
EVERY_REPORT = ["ITC-FY25#0001", "RELIANCE-FY25#0001", "TATAMOTORS-FY25#0001"]
SEARCHING = tool_turn(("s1", "search_reports", {"query": "rural demand", "report": "all"}))
# Room for one turn's worst case and not a second, once the first has been billed.
ONE_TURN_USD = worst_case_usd(MODEL, FAKE_USAGE.input_tokens, MAX_TOKENS_TOOL_LOOP, cached=True)


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
def test_the_model_s_search_runs_where_it_asked(
    make_llm: MakeLlm, search: FakeSearch, report: str, scope: str | None, found: list[str]
) -> None:
    asked = tool_turn(("s1", "search_reports", {"query": "rural demand", "report": report}))
    llm, fake = make_llm(turns(asked, ANSWER))

    result = run(llm, search, "who is most exposed to rural demand?")

    assert search.asked == [("rural demand", scope)]
    [searched] = result.searches
    assert (searched.query, searched.report, searched.found) == ("rural demand", report, found)
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
    ("limit_usd", "stopped", "turns", "searches"),
    [
        pytest.param(
            1.0,
            f"after {MAX_AGENT_TURNS} turns",
            MAX_AGENT_TURNS,
            MAX_AGENT_TURNS - 1,
            id="the turn cap, whose last turn's tools never run",
        ),
        pytest.param(
            ONE_TURN_USD + 0.001, "could cost up to", 1, 1, id="a turn the budget cannot afford"
        ),
    ],
)
def test_a_stopped_run_still_shows_what_it_searched(
    make_llm: MakeLlm,
    search: FakeSearch,
    limit_usd: float,
    stopped: str,
    turns: int,
    searches: int,
) -> None:
    llm, _ = make_llm(lambda sent: SEARCHING, limit_usd=limit_usd)

    result = run(llm, search, "who is most exposed to rural demand?")

    assert result.answer is None and stopped in str(result.stopped), result.stopped
    assert (result.turns, len(result.searches)) == (turns, searches)
    assert result.searches[0].found == EVERY_REPORT


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
