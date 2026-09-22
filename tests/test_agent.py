"""The first agent, on the SDK's real tool runner over the fake client: the model picks searches.

What is tested is the part that is ours — which report a search runs on, what the model is handed
back, and that thinking stays on. Whether the model searches well is a question for a live run.
"""

from __future__ import annotations

import pytest

from entropic.agent import run

from .conftest import FakeSearch, MakeLlm, tool_results, tool_turn, turns

ANSWER = "Tata Motors, on its small commercial vehicles [TATAMOTORS-FY25#0001]"


@pytest.mark.parametrize(
    ("report", "scope"),
    [
        pytest.param("TATAMOTORS-FY25", "TATAMOTORS-FY25", id="a named report is searched alone"),
        pytest.param("all", None, id="all searches every report"),
    ],
)
def test_the_model_s_search_runs_where_it_asked(
    make_llm: MakeLlm, search: FakeSearch, report: str, scope: str | None
) -> None:
    asked = tool_turn(("s1", "search_reports", {"query": "rural demand", "report": report}))
    llm, fake = make_llm(turns(asked, ANSWER))

    result = run(llm, search, "who is most exposed to rural demand?")

    assert search.asked == [("rural demand", scope)]
    assert result.searches == [("rural demand", report)]
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


def test_the_agent_keeps_thinking_on(make_llm: MakeLlm, search: FakeSearch) -> None:
    """The workflows switch thinking off; a tool-using agent must not, because Opus 5 without it
    can write a tool call into its text, where it never runs."""
    llm, fake = make_llm(lambda sent: "an answer")

    run(llm, search, "anything")

    assert fake.sent[0].thinking == {"type": "adaptive"}
