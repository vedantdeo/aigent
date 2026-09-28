"""Project 2's task set as an asset, and one task through the agent on the fake client."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("langgraph")

from aigent.agent import TOOLS  # noqa: E402
from aigent.agent_tasks import (  # noqa: E402
    DATASET,
    agent_task,
    graders,
    load_rows,
    recording,
    replayed,
    save_rows,
)
from aigent.evals.dataset import Case, load_jsonl  # noqa: E402
from aigent.evals.grade import Outcome  # noqa: E402
from aigent.tools_config import SANDBOX  # noqa: E402

from .conftest import FAKE_USAGE, FakeSearch, MakeLlm, tool_turn, turns  # noqa: E402

CASES = load_jsonl(DATASET)
NAMES = {str(tool["name"]) for tool in TOOLS}
MISSING_ON_PURPOSE = "tasks/board-minutes.txt"


def _names(case: Case, key: str) -> list[str]:
    return cast(list[str], case.expected.get(key, []))


def test_there_are_twenty_five_tasks() -> None:
    assert len(CASES) == 25


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_a_task_names_real_tools_and_orders_only_what_it_requires(case: Case) -> None:
    required, forbidden = _names(case, "tools"), _names(case, "forbid")
    assert set(required) <= NAMES and set(forbidden) <= NAMES, case.id
    assert not set(required) & set(forbidden), "a tool both required and forbidden"
    for pair in cast(list[list[str]], case.expected.get("order", [])):
        assert len(pair) == 2 and set(pair) <= set(required), pair
    assert str(case.expected.get("reference", "")).strip(), "the judge needs a reference"
    assert case.tags, "tagged by family"


@pytest.mark.parametrize(
    "case", [c for c in CASES if "read_file" in _names(c, "tools")], ids=lambda c: c.id
)
def test_a_file_a_task_reads_is_in_the_sandbox_unless_missing_on_purpose(case: Case) -> None:
    task = str(case.input["task"])
    named = next(word.rstrip(".,") for word in task.split() if word.startswith("tasks/"))
    assert (SANDBOX / named).exists() is (named != MISSING_ON_PURPOSE), named


def test_a_task_reports_its_tools_answer_and_whole_cost(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, fake = make_llm(
        turns(tool_turn(("s1", "search_reports", {"query": "q", "report": "all"})), "done")
    )
    case = Case.model_validate({"id": "pt-x", "input": {"task": "find it"}})

    outcome = agent_task(search, sdk=llm.sdk)(case)

    assert outcome.output["tools"] == ["search_reports"]
    assert outcome.output["finished"] is True and outcome.raw == "done"
    assert outcome.usage is not None
    assert outcome.usage.input_tokens == 2 * FAKE_USAGE.input_tokens, "both turns billed as one row"
    assert len(fake.sent) == 2


def test_a_task_that_fails_midway_still_reports_what_it_spent(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    """The first live run lost four rows' spending this way: billed by the API, absent from ours."""
    first = tool_turn(("s1", "search_reports", {"query": "q", "report": "all"}))
    llm, _ = make_llm(turns(first, RuntimeError("the second turn is refused")))
    case = Case.model_validate({"id": "pt-x", "input": {"task": "find it"}})

    outcome = agent_task(search, sdk=llm.sdk)(case)

    assert outcome.error is not None and "refused" in outcome.error
    assert outcome.usage is not None and outcome.usage.input_tokens == FAKE_USAGE.input_tokens


def test_a_saved_run_replays_its_answers_without_billing_them_again(
    tmp_path: Path, make_llm: MakeLlm, search: FakeSearch
) -> None:
    """Re-grading reads the answers back, and only the judge may spend."""
    llm, _ = make_llm(
        turns(tool_turn(("s1", "search_reports", {"query": "q", "report": "all"})), "done")
    )
    case = Case.model_validate({"id": "pt-x", "input": {"task": "find it"}})
    outcomes: dict[str, Outcome] = {}
    recording(agent_task(search, sdk=llm.sdk), outcomes)(case)
    path = tmp_path / "run.rows.jsonl"

    save_rows(path, outcomes)
    saved = load_rows(path)
    replay = replayed(saved)(case)

    assert replay.output == outcomes["pt-x"].output and replay.raw == "done"
    assert replay.usage is None, "the agent was billed once, when it ran"
    assert saved["pt-x"][1] == outcomes["pt-x"].usage, "its usage kept for the record"
    missing = replayed(saved)(Case.model_validate({"id": "pt-y", "input": {"task": "?"}}))
    assert missing.error is not None and "pt-y" in missing.error


def test_the_judge_is_told_todays_date() -> None:
    """Without it, a judge trained earlier reads a web answer's current figures as invented."""
    from datetime import date

    from aigent.evals.judge import LlmJudge

    judge = graders(today=date(2026, 9, 28))["correct"]

    assert isinstance(judge, LlmJudge) and judge.rubric.endswith("Today's date is 2026-09-28.")


@pytest.mark.parametrize(
    "agent",
    [
        pytest.param("sdk", id="the SDK's tool runner"),
        pytest.param("langgraph", id="LangGraph"),
        pytest.param("adk", id="Google's ADK"),
    ],
)
def test_every_build_reports_the_same_run_the_same_way(
    make_llm: MakeLlm, search: FakeSearch, agent: str
) -> None:
    """One scripted conversation through each build: the trajectory the graders read must not
    depend on which framework drove the loop."""
    pytest.importorskip("google.adk") if agent == "adk" else None
    both = tool_turn(
        ("s1", "search_reports", {"query": "q", "report": "all"}),
        ("c1", "calculate", {"expression": "1 + 1"}),
    )
    llm, _ = make_llm(turns(both, "two"))
    case = Case.model_validate({"id": "pt-x", "input": {"task": "find it, then add"}})

    outcome = agent_task(search, sdk=llm.sdk, agent=agent)(case)

    assert outcome.error is None, outcome.error
    assert outcome.output["tools"] == ["search_reports", "calculate"]
    assert outcome.output["finished"] is True and outcome.raw == "two"
