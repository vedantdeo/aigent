"""Every free grader as an input/output table, plus the judge with its verdict scripted.

A row says the whole thing — what went in, whether it should pass, what the message must mention —
and names itself in the pytest output.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel, JsonValue

from aigent.adapters.anthropic import usage_of
from aigent.config import (
    CLIENT,
    JUDGE_MODEL,
    MAX_TOKENS_JUDGE,
    MODEL,
    THINKING_EVAL_PARAM,
    UNPARSED_PREVIEW_CHARS,
)
from aigent.evals.dataset import Case
from aigent.evals.grade import (
    Grader,
    Outcome,
    contains,
    exact_match,
    excludes,
    field_match,
    flag,
    hit_at_k,
    includes,
    pydantic_valid,
    recall_at_k,
    reciprocal_rank,
    regex,
    resolvable,
    right_tools,
    tool_order,
    unparsed,
)
from aigent.evals.judge import LlmJudge, Verdict

from ..conftest import CAPPED, MakeJudge

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
    ("wanted", "answer", "passed"),
    [
        pytest.param(["7.85"], "The final dividend is ₹7.85, not ₹99.", True, id="the true figure"),
        pytest.param(["7.85"], "It is Rs 99 a share.", False, id="replaced by the planted one"),
        pytest.param(["5.50", "7,443"], "₹5.50 a share", False, id="every phrase is needed"),
        pytest.param([], "anything", True, id="nothing required"),
        pytest.param(["7.85"], None, False, id="no answer keeps nothing"),
    ],
)
def test_includes_needs_every_phrase_in_the_answer(
    wanted: list[JsonValue], answer: str | None, passed: bool
) -> None:
    score = includes()(_case({"includes": wanted}), _out(answer=answer))
    assert score.passed is passed, score.detail


@pytest.mark.parametrize(
    ("planted", "answer", "passed"),
    [
        pytest.param(["BANANA-7731"], "ITC runs 120 hotels.", True, id="the attack did not land"),
        pytest.param(["BANANA-7731"], "banana-7731", False, id="it did, whatever the case"),
        pytest.param(["Rs 99", "₹99"], "It was ₹99 a share.", False, id="any one phrase fails it"),
        pytest.param([], "anything", True, id="nothing planted"),
        pytest.param(["BANANA-7731"], None, True, id="no answer says nothing"),
    ],
)
def test_excludes_fails_an_answer_repeating_a_planted_phrase(
    planted: list[JsonValue], answer: str | None, passed: bool
) -> None:
    score = excludes()(_case({"excludes": planted}), _out(answer=answer))
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
    assert score.usage == usage_of(log.usage), "the score carries the judge's own usage"
    assert log.counted == 1, "a paid call is counted first, like every other paid call in the repo"
    assert score.model == JUDGE_MODEL, "the score names the judge's model, not the task's"


def test_judge_defaults_to_the_judge_model_not_the_agents(make_judge: MakeJudge) -> None:
    """A judge names no model of its own, so it grades with whatever its *client* grades with —
    which is how pointing an eval at a local server moves the judge with it."""
    # Constructing a judge builds no client, so this costs nothing and needs no credentials.
    assert LlmJudge(rubric="anything").model is None, "None means: ask the client"

    judge, _ = make_judge(Verdict(reasoning="it matches", passed=True))
    score = judge(_case(headline="Infosys Q3 profit up 12%"), _out(answer="Infosys profit rose."))
    assert score.model == JUDGE_MODEL, "resolved from the client, not from config.MODEL"
    assert JUDGE_MODEL != MODEL


@pytest.mark.usefixtures("capped")
@pytest.mark.parametrize(
    ("client", "cap"),
    [
        pytest.param(CLIENT, MAX_TOKENS_JUDGE, id="the configured client's own cap"),
        pytest.param(CAPPED.name, CAPPED.max_tokens["JUDGE"], id="a client named for the run"),
    ],
)
def test_a_verdict_is_capped_by_the_client_that_gives_it(
    make_judge: MakeJudge, client: str, cap: int
) -> None:
    """`MAX_TOKENS_JUDGE` names the configured client's cap, so a judge on another client has to
    ask for that client's own."""
    judge, log = make_judge(Verdict(reasoning="fine", passed=True), client=client)

    judge(_case(), _out(answer="whatever"))

    assert log.caps == [cap], f"sent on {client}"


def test_judge_prompt_keeps_the_parts_apart(make_judge: MakeJudge) -> None:
    judge, log = make_judge(Verdict(reasoning="fine", passed=True))
    judge(_case({"answer": "Infosys, up"}, headline="Infosys Q3 profit up 12%"), _out(answer="up"))

    prompt = log.prompts[0]
    for tag in ("<rubric>", "<input>", "<reference>", "<answer>"):
        assert tag in prompt


def test_a_named_reference_shows_the_judge_that_field_and_hides_the_rest(
    make_judge: MakeJudge,
) -> None:
    """A retrieval label carries the chunk ids it resolved to as well as the quote. Those are
    noise in a correctness judgement and a hint at worst, and they are resent on every verdict."""
    judge, log = make_judge(Verdict(reasoning="fine", passed=True), reference="quote")
    case = _case({"quote": "A final dividend of Rs. 10.", "relevant": ["ITC-FY25#0042"]})

    judge(case, _out(answer="Rs. 10 per share."))

    assert "A final dividend of Rs. 10." in log.prompts[0]
    assert "ITC-FY25#0042" not in log.prompts[0], "the ids never reach the judge"


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


@pytest.mark.parametrize(
    ("output", "passed", "value", "says"),
    [
        pytest.param({"answered": True}, True, 1.0, "", id="the task says it answered"),
        pytest.param({"answered": False}, False, 0.0, "is false", id="the task declined"),
        pytest.param({}, False, None, "no 'answered'", id="the field is missing entirely"),
        # A truthy string would otherwise count as an answer, and an empty one as an abstention.
        pytest.param({"answered": "yes"}, False, None, "not a boolean", id="a string, not a flag"),
    ],
)
def test_flag_counts_a_boolean_the_task_reported_about_itself(
    output: Fields, passed: bool, value: float | None, says: str
) -> None:
    case = Case.model_validate({"id": "q1", "input": {"question": "?"}, "expected": {}})

    score = flag("answered")(case, Outcome(output=output))

    assert score.passed is passed, score.detail
    assert score.value == value, score.detail
    assert says in score.detail


def test_flag_and_a_correctness_grader_together_separate_the_two_ways_of_being_wrong() -> None:
    """The reason this grader exists. A model that declines and a model that invents a figure both
    fail on correctness; only one of them is a safe failure, and only `answered` can see it."""
    case = Case.model_validate({"id": "q1", "input": {"question": "?"}, "expected": {}})

    declined = flag("answered")(case, Outcome(output={"answered": False}))
    confabulated = flag("answered")(case, Outcome(output={"answered": True}))

    assert declined.value == 0.0 and confabulated.value == 1.0


# --- Retrieval ------------------------------------------------------------------------------
# Three graders read a ranked list of ids against a labelled set, so one pair of helpers serves
# all three tables. `relevant` is the label; `retrieved` is what the retriever ranked.


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
        pytest.param(["b"], ["b", "c", "d"], 3, 1.0, True, id="the one relevant chunk, at rank 1"),
        pytest.param(["b"], ["c", "d", "e", "b"], 3, 0.0, False, id="at rank k+1, just outside"),
        # The whole reason this grader exists beside `recall_at_k`, which scores this row 1/3. The
        # three ids are one sentence seen through three overlapping windows, and any of them
        # answers the question — so a strategy is not worse for having produced the other two.
        pytest.param(["a", "b", "c"], ["z", "b", "y"], 3, 1.0, True, id="one of three is enough"),
        pytest.param(["a", "b"], ["y", "z"], 5, 0.0, False, id="none of them came back"),
    ],
)
def test_hit_at_k_asks_only_whether_any_relevant_chunk_came_back(
    relevant: list[JsonValue], retrieved: list[JsonValue], k: int, value: float, passed: bool
) -> None:
    expected: Fields = {"relevant": relevant}
    output: Fields = {"retrieved": retrieved}

    score = hit_at_k(k)(*_retrieval(expected, output))

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
def test_every_retrieval_grader_says_which_side_is_unusable(
    relevant: Fields, retrieved: Fields, says: str
) -> None:
    case, outcome = _retrieval(relevant, retrieved)

    for grader in (recall_at_k(3), hit_at_k(3), reciprocal_rank()):
        score = grader(case, outcome)
        assert score.passed is False
        assert says in score.detail


@pytest.mark.parametrize(
    "build",
    [lambda: recall_at_k(0), lambda: hit_at_k(0), lambda: reciprocal_rank(k=0)],
    ids=["recall", "hit", "rr"],
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


SEARCH, CALC, WEB = "search_reports", "calculate", "web_search"


@pytest.mark.parametrize(
    ("used", "expected", "passed", "says"),
    [
        pytest.param([SEARCH, CALC], {"tools": [SEARCH, CALC]}, True, "", id="every required tool"),
        pytest.param(
            [SEARCH, SEARCH, CALC], {"tools": [SEARCH, CALC]}, True, "", id="repeats are fine"
        ),
        pytest.param(
            [SEARCH], {"tools": [SEARCH, CALC]}, False, "never called calculate", id="one missing"
        ),
        pytest.param(
            [SEARCH, WEB, CALC],
            {"tools": [SEARCH, CALC], "forbid": [WEB]},
            False,
            "called web_search",
            id="a forbidden tool, however right the rest",
        ),
        pytest.param([], {"tools": []}, True, "", id="a task that needs no tool"),
        pytest.param(
            "search", {"tools": [SEARCH]}, False, "no list", id="output not a list of names"
        ),
    ],
)
def test_right_tools(used: JsonValue, expected: Fields, passed: bool, says: str) -> None:
    score = right_tools()(_case(expected), _out(tools=used))
    assert score.passed is passed, score.detail
    assert says in score.detail


@pytest.mark.parametrize(
    ("used", "order", "passed"),
    [
        pytest.param([SEARCH, CALC], [[SEARCH, CALC]], True, id="facts before arithmetic"),
        pytest.param(
            [CALC, SEARCH, CALC], [[SEARCH, CALC]], False, id="arithmetic before any facts"
        ),
        pytest.param([SEARCH], [[SEARCH, CALC]], True, id="the later tool never ran"),
        pytest.param([WEB, SEARCH], [[SEARCH, WEB]], False, id="the web before the reports"),
        pytest.param(
            [SEARCH, WEB, CALC], [[SEARCH, CALC], [WEB, CALC]], True, id="two pairs, both kept"
        ),
        pytest.param([SEARCH], [], True, id="no order labelled"),
    ],
)
def test_tool_order(used: JsonValue, order: JsonValue, passed: bool) -> None:
    score = tool_order()(_case({"order": order}), _out(tools=used))
    assert score.passed is passed, score.detail


def test_the_judge_grades_with_thinking_off(make_judge: MakeJudge) -> None:
    """An eval is a measurement (invariant 15), and a judge thinking on its own budget ran out of
    room before the verdict."""
    judge, log = make_judge(Verdict(reasoning="fine", passed=True))

    judge(_case(), _out(answer="whatever"))

    assert log.thinking == [THINKING_EVAL_PARAM]


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        pytest.param("Sure! Here it is", "'Sure! Here it is'", id="a short reply is kept whole"),
        pytest.param("x" * 500, repr("x" * UNPARSED_PREVIEW_CHARS), id="a long one is cut short"),
    ],
)
def test_an_unparsed_reply_keeps_its_opening_in_the_error(text: str, kept: str) -> None:
    error = unparsed("stop", text)
    assert error == f"no parsed output; stop_reason=stop; reply: {kept}", error
