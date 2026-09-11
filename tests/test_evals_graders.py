"""The four free graders as input/output tables, plus the judge with its verdict scripted.

Everything here runs without a network call, which is the point: graders are where an eval's
opinions live, and opinions you cannot test cheaply do not get tested.

A table is the shape these want. Every free grader is a pure function of (case, outcome), so a row
says the whole thing — what went in, whether it should pass, what the message has to mention — and
a failing row names itself in the pytest output, which a five-assert function cannot.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, JsonValue

from entropic.config import JUDGE_MODEL, MODEL
from entropic.evals.dataset import Case
from entropic.evals.grade import Outcome, contains, exact_match, field_match, pydantic_valid, regex
from entropic.evals.judge import LlmJudge, Verdict

from .conftest import MakeJudge

Fields = dict[str, JsonValue]


def _case(expected: Fields | None = None, **input_fields: object) -> Case:
    return Case.model_validate({"id": "c1", "input": input_fields, "expected": expected or {}})


def _out(**output: JsonValue) -> Outcome:
    return Outcome(output=output)


@pytest.mark.parametrize(
    ("expected", "output", "passed", "detail"),
    [
        ({"company": "Infosys"}, {"company": "  infosys "}, True, ()),
        ({"company": "Infosys"}, {"company": "Wipro"}, False, ("Infosys", "Wipro")),
        # Both of the next two fail, but one is a dataset bug and one a model failure. Say which.
        ({}, {"company": "Infosys"}, False, ("dataset",)),
        ({"company": "Infosys"}, {}, False, ("output",)),
    ],
)
def test_exact_match_normalises_case_and_whitespace_and_nothing_else(
    expected: Fields, output: Fields, passed: bool, detail: tuple[str, ...]
) -> None:
    score = exact_match("company")(_case(expected), Outcome(output=output))

    assert score.passed is passed, score.detail
    for fragment in detail:
        assert fragment in score.detail


@pytest.mark.parametrize(
    ("answer", "passed"),
    [
        ("It is worth 535,628 rupees.", True),
        # Literal, not numeric: the same number formatted differently does not match. Worth knowing
        # before you blame the model for a row `contains` scored as wrong.
        ("It is worth 535628 rupees.", False),
    ],
)
def test_contains_looks_for_the_label_inside_the_answer(answer: str, passed: bool) -> None:
    score = contains("answer")(_case({"answer": "535,628"}), _out(answer=answer))
    assert score.passed is passed, score.detail


@pytest.mark.parametrize(
    ("pattern", "value", "passed"),
    [
        (r"^[A-Z]{2,10}$", "INFY", True),
        (r"^[A-Z]{2,10}$", "INFY 500", False),
        (r"^[A-Z]+$", "INFY", True),
        (r"^[A-Z]+$", "infy", False),  # case must stay checkable
        (r"^\S+ \S+$", "two words", True),
        (r"^\S+ \S+$", "two  words", False),  # so must spacing
        (r"(?i)^[a-z]+$", "INFY", True),  # insensitivity is the pattern's call, not the grader's
        (r"^\d+$", 42, True),  # a non-string value is matched as its text form
    ],
)
def test_regex_matches_the_value_as_written(pattern: str, value: JsonValue, passed: bool) -> None:
    """Neither side is normalised, unlike exact_match. A format check has to be able to fail.

    Casefolding the subject would make `^[A-Z]+$` unsatisfiable; casefolding the pattern would be
    worse still, since `\\D` casefolds to `\\d` and silently inverts the check.
    """
    assert regex("v", pattern=pattern)(_case(), _out(v=value)).passed is passed


def test_regex_can_take_its_pattern_from_each_case() -> None:
    """How you check a format that varies per row — and what happens when that pattern is junk."""
    per_case = regex("date")
    assert per_case(_case({"date": r"\d{4}-\d{2}-\d{2}"}), _out(date="2026-09-10")).passed

    broken = per_case(_case({"date": "([unclosed"}), _out(date="2026-09-10"))
    assert not broken.passed
    assert "invalid regex" in broken.detail


class Extracted(BaseModel):
    company: str
    revenue_crore: float


def test_pydantic_valid_checks_shape_and_says_which_field_broke() -> None:
    grade = pydantic_valid(Extracted)
    assert grade(_case(), _out(company="Infosys", revenue_crore=39_000.0)).passed

    score = grade(_case(), _out(company="Infosys", revenue_crore="lots"))
    assert not score.passed
    assert "revenue_crore" in score.detail


@pytest.mark.parametrize(
    ("output", "why"),
    [
        ({"company": "Infosys", "quarter": "Q2"}, "a wrong field"),
        ({"company": "Infosys"}, "a missing field, which counts as wrong rather than as absent"),
    ],
)
def test_field_match_reports_each_field_even_when_the_record_fails(
    output: Fields, why: str
) -> None:
    score = field_match(["company", "quarter"])(
        _case({"company": "Infosys", "quarter": "Q3"}), Outcome(output=output)
    )

    assert not score.passed, f"{why} fails the record"
    assert score.parts == {"company": True, "quarter": False}
    assert "quarter" in score.detail


# --- the judge, with the model's verdict scripted -----------------------------------------------
# Not in the tables above: the judge takes a client, and what is worth asserting is the call it
# makes, not only the verdict it returns.


def test_judge_returns_the_verdict_and_bills_its_usage_to_the_score(make_judge: MakeJudge) -> None:
    judge, log = make_judge(Verdict(reasoning="Names Infosys and says profit rose.", passed=True))
    score = judge(_case(headline="Infosys Q3 profit up 12%"), _out(answer="Infosys profit rose."))

    assert score.passed
    assert score.detail.startswith("Names Infosys")
    assert score.usage == log.usage, "the score carries the judge's own usage, unmodified"
    assert log.counted == 1, "a paid call is counted first, like every other paid call in the repo"
    assert score.model == JUDGE_MODEL, "the score names the judge's model, not the task's"


def test_judge_defaults_to_the_judge_model_not_the_agents() -> None:
    # Constructing a judge builds no client, so this costs nothing and needs no credentials.
    assert LlmJudge(rubric="anything").model == JUDGE_MODEL
    assert JUDGE_MODEL != MODEL


def test_judge_prompt_keeps_the_parts_apart(make_judge: MakeJudge) -> None:
    judge, log = make_judge(Verdict(reasoning="fine", passed=True))
    judge(_case({"answer": "Infosys, up"}, headline="Infosys Q3 profit up 12%"), _out(answer="up"))

    prompt = log.prompts[0]
    for tag in ("<rubric>", "<input>", "<reference>", "<answer>"):
        assert tag in prompt


def test_judge_fails_closed_when_the_model_returns_nothing_parseable(make_judge: MakeJudge) -> None:
    judge, _ = make_judge(None)
    score = judge(_case(), _out(answer="whatever"))

    assert not score.passed
    assert "no verdict" in score.detail


def test_judge_does_not_spend_on_a_row_the_task_already_failed(make_judge: MakeJudge) -> None:
    judge, log = make_judge(Verdict(reasoning="unused", passed=True))
    score = judge(_case(), Outcome(error="429 rate limited"))

    assert not score.passed
    assert log.counted == 0, "no call at all for a row with no answer to grade"
