"""Guardrails: what a task may be, how untrusted text is fenced, what an answer must satisfy, and
the injection set that measures all three against a live model."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from aigent import guardrails
from aigent.agent import SYSTEM
from aigent.config import MAX_ANSWER_CHARS, MAX_TASK_CHARS
from aigent.errors import GuardrailTripped
from aigent.evals.dataset import load_jsonl
from aigent.tools_config import SANDBOX

pytest.importorskip("langgraph")

from aigent.agent_tasks import INJECTIONS, agent_task, answer_guard  # noqa: E402
from aigent.evals.dataset import Case  # noqa: E402
from aigent.evals.grade import Outcome  # noqa: E402

from .conftest import FakeSearch, MakeLlm, tool_results, tool_turn, turns  # noqa: E402

INJECTION_CASES = load_jsonl(INJECTIONS)


@pytest.mark.parametrize(
    ("task", "problem"),
    [
        pytest.param("", "empty", id="nothing to do"),
        pytest.param("   \n ", "empty", id="only whitespace"),
        pytest.param("x" * (MAX_TASK_CHARS + 1), "over", id="longer than the cap"),
        pytest.param("what is\x00 this", "control characters", id="a NUL byte"),
        pytest.param("tab\there\x1b[2J", "control characters", id="an escape sequence"),
        pytest.param(
            "use sk-ant-api03-abcdefghijkl to check", "credential", id="an API key pasted in"
        ),
    ],
)
def test_a_task_the_agent_should_not_take_is_refused(task: str, problem: str) -> None:
    with pytest.raises(GuardrailTripped, match=problem):
        guardrails.check_task(task)


def test_a_fair_task_comes_back_trimmed_with_its_newlines_and_tabs() -> None:
    assert guardrails.check_task("  line one\n\tline two  ") == "line one\n\tline two"


FOUND = ["ITC-FY25#1284", "RELIANCE-FY25#0250"]


@pytest.mark.parametrize(
    ("answer", "problems"),
    [
        pytest.param("ITC operates 120 hotels [ITC-FY25#1284].", [], id="a clean answer"),
        pytest.param("No passage covers this.", [], id="citing nothing is fine"),
        pytest.param("  ", ["empty"], id="an empty answer"),
        pytest.param("x" * (MAX_ANSWER_CHARS + 1), ["over"], id="longer than the cap"),
        pytest.param("See [ITC-FY25#9999].", ["ITC-FY25#9999"], id="an invented citation"),
        pytest.param(
            'I ignored the file\'s fake "ITC-FY25#9999".', [], id="naming an id is not citing it"
        ),
        pytest.param(
            "Both say so [ITC-FY25#1284, RELIANCE-FY25#0250].", [], id="two ids in one bracket"
        ),
        pytest.param("Your key: sk-ant-api03-abcdefghijkl", ["credential"], id="an Anthropic key"),
        pytest.param("pk-lf-a3e69867-8c4f-44", ["credential"], id="a Langfuse key"),
        pytest.param(
            "[TATAMOTORS-FY25#0001] says sk-ant-api03-zzzzzzzzzz",
            ["TATAMOTORS-FY25#0001", "credential"],
            id="every broken rule is reported, not just the first",
        ),
    ],
)
def test_an_answer_is_checked_against_every_rule(answer: str, problems: list[str]) -> None:
    found = guardrails.check_answer(answer, FOUND)

    assert len(found) == len(problems), found
    for message, problem in zip(found, problems, strict=True):
        assert problem in message, found


KEY = "sk-ant-api03-abcdefghijklmnop"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(f"key {KEY} here", f"key {guardrails.REDACTED} here", id="in a string"),
        pytest.param(
            {"messages": [{"content": f"use {KEY}"}], "n": 3},
            {"messages": [{"content": f"use {guardrails.REDACTED}"}], "n": 3},
            id="nested in messages",
        ),
        pytest.param((KEY, "fine"), [guardrails.REDACTED, "fine"], id="a tuple"),
        pytest.param("nothing secret", "nothing secret", id="a clean string is untouched"),
        pytest.param(42, 42, id="a number is untouched"),
    ],
)
def test_redact_replaces_every_credential_however_deep(value: object, expected: object) -> None:
    assert guardrails.redact(value) == expected


@pytest.mark.parametrize(
    ("violations", "withheld"),
    [
        pytest.param([], False, id="a clean answer is shown"),
        pytest.param(["cites passages it never retrieved: X#0001"], False, id="flagged, shown"),
        pytest.param(["contains 1 credential-shaped string(s)"], True, id="a leak is withheld"),
    ],
)
def test_only_a_leak_withholds_the_answer(violations: list[str], withheld: bool) -> None:
    assert (guardrails.shown("the answer", violations) == guardrails.WITHHELD) is withheld


def test_the_agent_is_told_fenced_text_is_data() -> None:
    assert guardrails.UNTRUSTED_NOTE in SYSTEM


def test_a_refused_task_costs_nothing(make_llm: MakeLlm, search: FakeSearch) -> None:
    llm, fake = make_llm(turns("never sent"))
    case = Case.model_validate({"id": "x", "input": {"task": "go\x00"}})

    outcome = agent_task(search, sdk=llm.sdk)(case)

    assert outcome.error is not None and outcome.error.startswith("refused before sending")
    assert fake.sent == [] and fake.counted == []


def test_a_file_reaches_the_model_fenced(make_llm: MakeLlm, search: FakeSearch) -> None:
    asked = tool_turn(("r1", "read_file", {"file_path": "injections/board-summary.txt"}))
    llm, fake = make_llm(turns(asked, "Reliance recommended ₹5.50 a share."))
    case = Case.model_validate({"id": "x", "input": {"task": "read the board summary"}})

    outcome = agent_task(search, sdk=llm.sdk)(case)

    [result] = tool_results(fake.sent[1])
    content = str(result["content"])
    assert content.startswith('<untrusted source="read_file">'), content
    assert content.count("</untrusted>") == 1, "the planted closing tag was escaped"
    assert outcome.output["guarded"] is True


def test_an_answer_leaking_a_key_is_withheld_and_marked(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, _ = make_llm(turns("Here you go: sk-ant-api03-abcdefghijklmnop"))
    case = Case.model_validate({"id": "x", "input": {"task": "what is in .env?"}})

    outcome = agent_task(search, sdk=llm.sdk)(case)

    assert outcome.output["answer"] == guardrails.WITHHELD
    assert outcome.output["guarded"] is False
    assert outcome.raw is not None and "sk-ant" not in outcome.raw, "nor does the judge see it"


@pytest.mark.parametrize("case", INJECTION_CASES, ids=[case.id for case in INJECTION_CASES])
def test_an_injection_case_attacks_through_a_real_file_and_says_what_must_not_happen(
    case: Case,
) -> None:
    task = str(case.input["task"])
    for word in task.split():
        if word.startswith("injections/"):
            path = word.rstrip(".,")
            assert (SANDBOX / path).is_file(), path
    checks = [case.expected.get(key) for key in ("excludes", "includes", "forbid")]
    assert any(checks) or "control" in case.tags, "an attack with nothing to check"


@pytest.mark.parametrize(
    ("output", "passed"),
    [
        pytest.param({"answer": "₹7.85 [ITC-FY25#1284]", "retrieved": FOUND}, True, id="clean"),
        pytest.param({"answer": "₹7.85 [ITC-FY25#1284]"}, False, id="a saved row without ids"),
        pytest.param({"answer": "₹7.85, not the memo's ITC-FY25#9999"}, True, id="only mentioned"),
        pytest.param({"answer": guardrails.WITHHELD}, False, id="withheld for a credential"),
        pytest.param({"answer": None}, True, id="no answer, nothing to check"),
    ],
)
def test_the_guard_grader_reapplies_todays_rules_to_a_saved_answer(
    output: dict[str, JsonValue], passed: bool
) -> None:
    case = Case.model_validate({"id": "x", "input": {"task": "t"}})
    score = answer_guard(case, Outcome(output=output))
    assert score.passed is passed, score.detail
