"""The four free graders, plus the judge with its verdict scripted.

Everything here runs without a network call, which is the point: graders are where an eval's
opinions live, and opinions you cannot test cheaply do not get tested.
"""

from __future__ import annotations

from typing import cast

import anthropic
from anthropic.types import MessageTokensCount, Usage
from pydantic import BaseModel, JsonValue

from entropic.config import JUDGE_MODEL, MODEL
from entropic.evals.dataset import Case
from entropic.evals.grade import Outcome, contains, exact_match, field_match, pydantic_valid, regex
from entropic.evals.judge import LlmJudge, Verdict


def _case(expected: dict[str, object] | None = None, **input_fields: object) -> Case:
    return Case.model_validate({"id": "c1", "input": input_fields, "expected": expected or {}})


def _out(**output: JsonValue) -> Outcome:
    return Outcome(output=output)


def test_exact_match_ignores_case_and_surrounding_whitespace() -> None:
    grade = exact_match("company")
    assert grade(_case({"company": "Infosys"}), _out(company="  infosys ")).passed


def test_exact_match_reports_both_sides_when_it_fails() -> None:
    score = exact_match("company")(_case({"company": "Infosys"}), _out(company="Wipro"))
    assert not score.passed
    assert "Infosys" in score.detail and "Wipro" in score.detail


def test_a_missing_label_reads_differently_from_a_missing_answer() -> None:
    # Both fail, but one is a dataset bug and the other is a model failure. Say which.
    no_label = exact_match("company")(_case({}), _out(company="Infosys"))
    no_answer = exact_match("company")(_case({"company": "Infosys"}), _out())
    assert "dataset" in no_label.detail
    assert "output" in no_answer.detail


def test_contains_looks_for_the_label_inside_the_answer() -> None:
    grade = contains("answer")
    case = _case({"answer": "535,628"})
    assert grade(case, _out(answer="It is worth 535,628 rupees.")).passed
    # Literal, not numeric: the same number formatted differently does not match. Worth knowing
    # before you blame the model for a row `contains` scored as wrong.
    assert not grade(case, _out(answer="It is worth 535628 rupees.")).passed


def test_regex_takes_a_fixed_pattern_or_one_per_case() -> None:
    fixed = regex("ticker", pattern=r"^[A-Z]{2,10}$")
    assert fixed(_case(), _out(ticker="INFY")).passed
    assert not fixed(_case(), _out(ticker="INFY 500")).passed

    per_case = regex("date")
    assert per_case(_case({"date": r"\d{4}-\d{2}-\d{2}"}), _out(date="2026-09-10")).passed


def test_regex_matches_the_value_as_written() -> None:
    """Neither side is normalised, unlike exact_match. A format check has to be able to fail.

    Casefolding the subject would make `^[A-Z]+$` unsatisfiable; casefolding the pattern would be
    worse still, since `\\D` casefolds to `\\d` and silently inverts the check.
    """
    upper = regex("ticker", pattern=r"^[A-Z]+$")
    assert upper(_case(), _out(ticker="INFY")).passed
    assert not upper(_case(), _out(ticker="infy")).passed, "case must be checkable"

    # Whitespace survives too, so a spacing rule means what it says.
    one_space = regex("answer", pattern=r"^\S+ \S+$")
    assert one_space(_case(), _out(answer="two words")).passed
    assert not one_space(_case(), _out(answer="two  words")).passed

    # Insensitivity is the pattern's decision to make, not the grader's.
    assert regex("ticker", pattern=r"(?i)^[a-z]+$")(_case(), _out(ticker="INFY")).passed


def test_regex_still_works_on_a_value_that_is_not_a_string() -> None:
    assert regex("count", pattern=r"^\d+$")(_case(), _out(count=42)).passed


def test_regex_says_so_when_the_pattern_itself_is_broken() -> None:
    score = regex("date")(_case({"date": "([unclosed"}), _out(date="2026-09-10"))
    assert not score.passed
    assert "invalid regex" in score.detail


class Extracted(BaseModel):
    company: str
    revenue_crore: float


def test_pydantic_valid_checks_shape_and_says_which_field_broke() -> None:
    grade = pydantic_valid(Extracted)
    assert grade(_case(), _out(company="Infosys", revenue_crore=39_000.0)).passed

    score = grade(_case(), _out(company="Infosys", revenue_crore="lots"))
    assert not score.passed
    assert "revenue_crore" in score.detail


def test_field_match_reports_each_field_even_when_the_record_fails() -> None:
    grade = field_match(["company", "quarter", "direction"])
    case = _case({"company": "Infosys", "quarter": "Q3", "direction": "up"})
    score = grade(case, _out(company="Infosys", quarter="Q2", direction="up"))

    assert not score.passed, "one wrong field fails the record"
    assert score.parts == {"company": True, "quarter": False, "direction": True}
    assert "quarter" in score.detail


def test_field_match_counts_a_missing_field_as_wrong_not_as_absent() -> None:
    score = field_match(["company", "quarter"])(
        _case({"company": "Infosys", "quarter": "Q3"}), _out(company="Infosys")
    )
    assert score.parts == {"company": True, "quarter": False}


# --- the judge, with the model's verdict scripted -----------------------------------------------


class _FakeParsed:
    def __init__(self, verdict: Verdict | None) -> None:
        self.parsed_output = verdict
        self.usage = Usage(input_tokens=120, output_tokens=40)
        self.stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self, verdict: Verdict | None) -> None:
        self._verdict = verdict
        self.prompts: list[str] = []
        self.counted = 0

    def count_tokens(self, **_: object) -> MessageTokensCount:
        self.counted += 1
        return MessageTokensCount(input_tokens=120)

    def parse(self, **kwargs: object) -> _FakeParsed:
        messages = cast("list[dict[str, str]]", kwargs["messages"])
        self.prompts.append(messages[0]["content"])
        return _FakeParsed(self._verdict)


class _FakeClient:
    def __init__(self, verdict: Verdict | None) -> None:
        self.messages = _FakeMessages(verdict)


def _judge(verdict: Verdict | None) -> tuple[LlmJudge, _FakeMessages]:
    fake = _FakeClient(verdict)
    judge = LlmJudge(
        rubric="The answer must name the company and the direction of the move.",
        client=cast(anthropic.Anthropic, fake),
    )
    return judge, fake.messages


def test_judge_returns_the_verdict_and_bills_its_usage_to_the_score() -> None:
    judge, log = _judge(Verdict(reasoning="Names Infosys and says profit rose.", passed=True))
    score = judge(_case(headline="Infosys Q3 profit up 12%"), _out(answer="Infosys profit rose."))

    assert score.passed
    assert score.detail.startswith("Names Infosys")
    assert score.usage is not None and score.usage.output_tokens == 40
    assert log.counted == 1, "a paid call is counted first, like every other paid call in the repo"
    assert score.model == JUDGE_MODEL, "the score names the judge's model, not the task's"


def test_judge_defaults_to_the_judge_model_not_the_agents() -> None:
    # Constructing a judge builds no client, so this costs nothing and needs no credentials.
    assert LlmJudge(rubric="anything").model == JUDGE_MODEL
    assert JUDGE_MODEL != MODEL


def test_judge_prompt_keeps_the_parts_apart() -> None:
    judge, log = _judge(Verdict(reasoning="fine", passed=True))
    judge(_case({"answer": "Infosys, up"}, headline="Infosys Q3 profit up 12%"), _out(answer="up"))

    prompt = log.prompts[0]
    for tag in ("<rubric>", "<input>", "<reference>", "<answer>"):
        assert tag in prompt


def test_judge_fails_closed_when_the_model_returns_nothing_parseable() -> None:
    judge, _ = _judge(None)
    score = judge(_case(), _out(answer="whatever"))
    assert not score.passed
    assert "no verdict" in score.detail


def test_judge_does_not_spend_on_a_row_the_task_already_failed() -> None:
    judge, log = _judge(Verdict(reasoning="unused", passed=True))
    score = judge(_case(), Outcome(error="429 rate limited"))

    assert not score.passed
    assert log.counted == 0, "no call at all for a row with no answer to grade"
