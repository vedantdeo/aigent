"""Project 2's task set as an asset, and one task through the agent on the fake client."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from anthropic.types import CitationsWebSearchResultLocation, TextBlock

pytest.importorskip("langgraph")

from aigent import agent_tasks  # noqa: E402
from aigent.agent import TOOLS  # noqa: E402
from aigent.agent_graph import Traced  # noqa: E402
from aigent.agent_tasks import (  # noqa: E402
    DATASET,
    agent_task,
    graders,
    judged_text,
    load_rows,
    main,
    recording,
    replayed,
    save_rows,
)
from aigent.evals.dataset import Case, load_jsonl  # noqa: E402
from aigent.evals.grade import Grader, Outcome  # noqa: E402
from aigent.evals.runner import EvalRun  # noqa: E402
from aigent.pricing import estimate_eval_usd  # noqa: E402
from aigent.tools_config import SANDBOX  # noqa: E402

from .conftest import (  # noqa: E402
    FAKE_USAGE,
    FakeSearch,
    MakeLlm,
    Recorder,
    tool_turn,
    turns,
)

CASES = load_jsonl(DATASET)
PAGE = "https://rates.example/aud-inr"
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
    asked = tool_turn(
        ("s1", "search_reports", {"query": "q", "report": "all"}),
        ("c1", "calculate", {"expression": "1 + 1"}),
    )
    citation = CitationsWebSearchResultLocation(
        type="web_search_result_location",
        url=PAGE,
        title="a page",
        encrypted_index="x",
        cited_text="about 58",
    )
    said = TextBlock(type="text", text="about 58", citations=[citation])
    both = asked.model_copy(update={"content": [said, *asked.content]})
    llm, _ = make_llm(turns(both, "two"))
    case = Case.model_validate({"id": "pt-x", "input": {"task": "find it, then add"}})

    outcome = agent_task(search, sdk=llm.sdk, agent=agent)(case)

    assert outcome.error is None, outcome.error
    assert outcome.output["tools"] == ["search_reports", "calculate"]
    assert outcome.output["finished"] is True
    assert outcome.output["sources"] == [PAGE], "a page cited a turn before the answer"
    assert outcome.raw == f"two\n\nWeb sources cited: {PAGE}", "the judge sees it"


@pytest.mark.parametrize("agent", ["sdk", "langgraph"])
def test_both_builds_trace_the_same_tree(
    make_llm: MakeLlm, search: FakeSearch, traces: Recorder, agent: str
) -> None:
    """An agent observation holding each turn's generation, with the tools it asked for between
    them: Langfuse's shape for an agent loop, whichever framework drove it."""
    asked = tool_turn(
        ("s1", "search_reports", {"query": "q", "report": "all"}),
        ("c1", "calculate", {"expression": "1 + 1"}),
    )
    llm, _ = make_llm(turns(asked, "two"))
    case = Case.model_validate({"id": "pt-x", "input": {"task": "find it, then add"}})

    agent_task(search, sdk=llm.sdk, agent=agent)(case)

    [run] = traces.roots
    assert (run.name, run.kind, run.inputs, run.output) == (
        "answer-task",
        "agent",
        "find it, then add",
        "two",
    )
    assert [(seen.name, seen.kind) for seen in run.children] == [
        ("agent", "generation"),
        ("search_reports", "retriever"),
        ("calculate", "tool"),
        ("agent", "generation"),
    ]


@pytest.mark.parametrize(
    ("answer", "sources", "judged"),
    [
        pytest.param(None, [PAGE], "(no answer)", id="no answer, whatever was cited"),
        pytest.param("58", [], "58", id="an answer citing nothing is read as written"),
        pytest.param(
            "58",
            [PAGE, PAGE + "/2"],
            f"58\n\nWeb sources cited: {PAGE}, {PAGE}/2",
            id="cited pages follow the answer, in order",
        ),
    ],
)
def test_the_judge_reads_the_answer_and_the_pages_it_cited(
    answer: str | None, sources: list[str], judged: str
) -> None:
    traced = Traced(answer, [], 1, None, [], sources)

    assert judged_text(traced) == judged


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        pytest.param(
            ["--model", "claude-opus-5", "--judge-model", "claude-opus-5"],
            "grade its own answers",
            id="a model judging itself",
        ),
        pytest.param(
            ["--judge-model", "claude-opus-5-5"],
            "claude-opus-5-5 has no price",
            id="a judge the dry run would price at nothing",
        ),
        pytest.param(
            ["--model", "claude-nope", "--judge-model", "claude-sonnet-5"],
            "claude-nope has no price",
            id="an agent model with no price",
        ),
    ],
)
def test_a_run_that_cannot_be_priced_or_judged_fairly_is_refused_before_the_estimate(
    argv: list[str], says: str
) -> None:
    with pytest.raises(SystemExit, match=says):
        main(argv)


def test_a_regrade_is_priced_for_the_answers_saved_not_the_whole_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A one-answer file once dry-ran at 25 judge calls' worst case."""
    path = tmp_path / "one.rows.jsonl"
    save_rows(
        path, {CASES[0].id: Outcome(output={"answer": "x"}, raw="x", model="claude-sonnet-5")}
    )
    one = estimate_eval_usd("claude-opus-5", 1, 2_000, 512)

    main(["--model", "claude-sonnet-5", "--judge-model", "claude-opus-5", "--regrade", str(path)])

    assert f"worst case ${one:.2f}" in capsys.readouterr().out


class _NoClient:
    """Stands in for `Llm` where `main` only wants a connection to hand on."""

    sdk = None

    @classmethod
    def for_eval(cls, client: str) -> _NoClient:
        del client
        return cls()


@pytest.mark.parametrize(
    ("argv", "workers", "graded_by"),
    [
        pytest.param(
            ["--set", "injections", "--only", "inj-001,inj-002", "--workers", "3"],
            3,
            {"right_tools", "includes", "excludes", "guarded", "finished"},
            id="the injection set, three at once, graded free",
        ),
        pytest.param(
            ["--only", "pt-001", "--model", "claude-sonnet-5", "--judge-model", "claude-opus-5"],
            1,
            {"right_tools", "tool_order", "finished", "correct"},
            id="the task set, serial by default, judged",
        ),
        pytest.param(
            [
                "--regrade",
                "{saved}",
                "--model",
                "claude-sonnet-5",
                "--judge-model",
                "claude-opus-5",
            ],
            2,
            {"right_tools", "tool_order", "finished", "correct"},
            id="a re-grade, two at once",
        ),
    ],
)
def test_a_paid_run_hands_its_workers_and_graders_to_the_runner(
    argv: list[str],
    workers: int,
    graded_by: set[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = tmp_path / "saved.rows.jsonl"
    save_rows(saved, {"pt-001": Outcome(output={"answer": "x"}, raw="x", model="claude-sonnet-5")})
    if "{saved}" in argv:
        argv = [str(saved) if arg == "{saved}" else arg for arg in argv] + ["--workers", "2"]
    handed: dict[str, Any] = {}

    def runner(
        cases: list[Case], variants: dict[str, object], graders: dict[str, Grader], **kw: Any
    ) -> EvalRun:
        handed.update(kw, cases=[case.id for case in cases], graders=set(graders))
        return EvalRun("d", "", "m", tuple(variants), tuple(graders))

    monkeypatch.setattr(agent_tasks, "run_eval", runner)
    monkeypatch.setattr(agent_tasks, "Llm", _NoClient)
    monkeypatch.setattr(agent_tasks, "build_search", lambda cache: FakeSearch())
    monkeypatch.setattr(agent_tasks, "write_report", lambda run: tmp_path / "report.md")
    monkeypatch.setattr(agent_tasks, "save_rows", lambda path, outcomes: None)

    main([*argv, "--yes"])

    assert handed["workers"] == workers, handed
    assert handed["graders"] == graded_by, handed["graders"]
    assert handed["worst_usd"] is not None, "every paid run is admitted whole"
