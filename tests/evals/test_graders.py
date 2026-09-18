"""Every free grader as an input/output table, plus the judge with its verdict scripted.

A row says the whole thing — what went in, whether it should pass, what the message must mention —
and names itself in the pytest output.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel, JsonValue

from entropic.config import JUDGE_MODEL, MODEL
from entropic.evals.dataset import Case
from entropic.evals.grade import (
    Grader,
    Outcome,
    contains,
    exact_match,
    field_match,
    pydantic_valid,
    recall_at_k,
    reciprocal_rank,
    regex,
    resolvable,
)
from entropic.evals.judge import LlmJudge, Verdict

from ..conftest import MakeJudge

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
    """Neither side is normalised: casefolding a pattern would turn `\\D` into `\\d`."""
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


# --- Retrieval ------------------------------------------------------------------------------
# Both graders read a ranked list of ids against a labelled set, so one pair of helpers serves both
# tables. `relevant` is the label; `retrieved` is what the retriever ranked.


def _retrieval(relevant: Fields, retrieved: Fields) -> tuple[Case, Outcome]:
    return (
        Case.model_validate({"id": "q1", "input": {"question": "?"}, "expected": relevant}),
        Outcome(output=retrieved),
    )


@pytest.mark.parametrize(
    ("relevant", "retrieved", "k", "value", "passed"),
    [
        pytest.param(["b"], ["b", "c", "d"], 3, 1.0, True, id="the one relevant chunk, at rank 1"),
        pytest.param(["b"], ["c", "d", "b"], 3, 1.0, True, id="at rank k, still inside"),
        pytest.param(["b"], ["c", "d", "e", "b"], 3, 0.0, False, id="at rank k+1, just outside"),
        pytest.param(["a", "b", "c"], ["a", "b", "z"], 3, 2 / 3, False, id="two of three found"),
        pytest.param(["a", "b"], ["a", "b"], 5, 1.0, True, id="fewer retrieved than k"),
        # A retriever that returns the same chunk five times has found one thing, not five. Without
        # the de-duplication in `_id_list`, `b` here would sit at rank 6 and score zero.
        pytest.param(["b"], ["a", "a", "a", "a", "a", "b"], 5, 1.0, True, id="repeats collapse"),
    ],
)
def test_recall_at_k_counts_the_relevant_chunks_inside_the_top_k(
    relevant: list[JsonValue], retrieved: list[JsonValue], k: int, value: float, passed: bool
) -> None:
    expected: Fields = {"relevant": relevant}
    output: Fields = {"retrieved": retrieved}

    score = recall_at_k(k)(*_retrieval(expected, output))

    assert score.passed is passed, score.detail
    assert score.value == pytest.approx(value), score.detail


@pytest.mark.parametrize(
    ("relevant", "retrieved", "k", "value", "passed"),
    [
        pytest.param(["b"], ["b", "c"], None, 1.0, True, id="first hit at rank 1"),
        pytest.param(["b"], ["a", "b"], None, 0.5, False, id="first hit at rank 2"),
        pytest.param(["b"], ["a", "c", "d", "b"], None, 0.25, False, id="first hit at rank 4"),
        pytest.param(["b"], ["a", "c"], None, 0.0, False, id="not retrieved at all"),
        # Rank is what decides whether a chunk survives the context window, so an MRR@k has to cut
        # the list where the generator will: found at rank 3 is not found when two are passed on.
        pytest.param(["b"], ["a", "c", "b"], 2, 0.0, False, id="found, but below the k handed on"),
        pytest.param(["y", "b"], ["a", "b", "y"], None, 0.5, False, id="the earliest of several"),
    ],
)
def test_reciprocal_rank_is_one_over_the_rank_of_the_first_relevant_chunk(
    relevant: list[JsonValue], retrieved: list[JsonValue], k: int | None, value: float, passed: bool
) -> None:
    expected: Fields = {"relevant": relevant}
    output: Fields = {"retrieved": retrieved}

    score = reciprocal_rank(k=k)(*_retrieval(expected, output))

    assert score.passed is passed, score.detail
    assert score.value == pytest.approx(value), score.detail


@pytest.mark.parametrize(
    ("relevant", "retrieved", "says"),
    [
        pytest.param({}, {"retrieved": ["a"]}, "dataset", id="no label on the case"),
        pytest.param({"relevant": "a"}, {"retrieved": ["a"]}, "list", id="label is a bare string"),
        pytest.param(
            {"relevant": []}, {"retrieved": ["a"]}, "no single chunk", id="label is empty"
        ),
        pytest.param({"relevant": ["a"]}, {}, "output", id="task returned no ranked list"),
        pytest.param({"relevant": ["a"]}, {"retrieved": [1, 2]}, "list", id="ids are not strings"),
    ],
)
def test_both_retrieval_graders_say_which_side_is_unusable(
    relevant: Fields, retrieved: Fields, says: str
) -> None:
    case, outcome = _retrieval(relevant, retrieved)

    for grader in (recall_at_k(3), reciprocal_rank()):
        score = grader(case, outcome)
        assert score.passed is False
        assert says in score.detail


@pytest.mark.parametrize(
    "build", [lambda: recall_at_k(0), lambda: reciprocal_rank(k=0)], ids=["recall", "rr"]
)
def test_a_k_below_one_is_refused_when_the_grader_is_built_not_when_it_runs(
    build: Callable[[], Grader],
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build()


@pytest.mark.parametrize(
    ("relevant", "survived", "value"),
    [
        pytest.param({"relevant": ["c#0001"]}, True, 1.0, id="the quote is inside one chunk"),
        pytest.param(
            {"relevant": ["c#0001", "c#0002"]}, True, 1.0, id="or inside two, which is still found"
        ),
        pytest.param({"relevant": []}, False, 0.0, id="the quote straddles a chunk boundary"),
        pytest.param({}, False, 0.0, id="the case carries no label at all"),
        pytest.param({"relevant": "c#0001"}, False, 0.0, id="the label is not a list"),
    ],
)
def test_resolvable_reports_whether_the_label_survived_chunking(
    relevant: Fields, survived: bool, value: float
) -> None:
    """A chunking property, graded separately so it stops being charged to the retriever."""
    case, outcome = _retrieval(relevant, {"retrieved": ["c#0001"]})

    score = resolvable()(case, outcome)

    assert score.passed is survived, score.detail
    assert score.value == value, score.detail


def test_resolvable_carries_a_number_where_the_other_two_carry_none() -> None:
    """The reason all three belong in one table: this one means over every case, they do not.

    `recall_at_k` and `reciprocal_rank` decline to score an unresolvable row, so `_metrics_table`
    leaves it out of their mean. Without `resolvable` beside them their columns look like whole
    dataset averages and are not.
    """
    case, outcome = _retrieval({"relevant": []}, {"retrieved": ["c#0001"]})

    assert resolvable()(case, outcome).value == 0.0
    assert recall_at_k(5)(case, outcome).value is None
    assert reciprocal_rank()(case, outcome).value is None
