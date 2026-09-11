"""The runner's three promises: a bad row is data, a tripped ceiling keeps what it paid for, and
every dollar spent lands in the total.

None of these tests touch the network. The task under test is a plain function, which is the whole
reason the harness takes one instead of a prompt.
"""

from __future__ import annotations

import pytest
from anthropic.types import Usage

from entropic.evals.dataset import Case
from entropic.evals.grade import Grader, Outcome, Score, exact_match
from entropic.evals.runner import Task, run_eval

MODEL = "claude-opus-5"
# $0.005 of input plus $0.025 of output on Opus.
COSTLY = Usage(input_tokens=1000, output_tokens=1000)


def _cases(n: int) -> list[Case]:
    return [
        Case.model_validate(
            {"id": f"c{i}", "input": {"q": f"question {i}"}, "expected": {"answer": "yes"}}
        )
        for i in range(1, n + 1)
    ]


def _always(answer: str, usage: Usage | None = None) -> Task:
    def task(case: Case) -> Outcome:
        del case
        return Outcome(output={"answer": answer}, usage=usage, model=MODEL)

    return task


def test_scores_every_case_and_reports_the_run(capsys: pytest.CaptureFixture[str]) -> None:
    run = run_eval(
        _cases(3),
        {"baseline": _always("yes")},
        {"exact": exact_match("answer")},
        dataset="toy.jsonl",
        digest="abc123",
    )

    assert len(run.rows) == 3
    assert all(row.passed for row in run.rows)
    assert run.spent_usd == 0.0, "a task with no usage spends nothing"
    assert run.stopped_early is None
    assert "3 rows" in capsys.readouterr().out


def test_variants_are_the_inner_loop_so_a_short_run_still_compares() -> None:
    run = run_eval(
        _cases(2),
        {"a": _always("yes"), "b": _always("no")},
        {"exact": exact_match("answer")},
        progress=False,
    )

    assert [(row.case_id, row.variant) for row in run.rows] == [
        ("c1", "a"),
        ("c1", "b"),
        ("c2", "a"),
        ("c2", "b"),
    ]
    assert [row.passed for row in run.rows_for("a")] == [True, True]
    assert [row.passed for row in run.rows_for("b")] == [False, False]


def test_a_task_that_raises_becomes_an_error_row_and_the_run_continues() -> None:
    def explodes_once(case: Case) -> Outcome:
        if case.id == "c2":
            raise RuntimeError("connection reset")
        return Outcome(output={"answer": "yes"})

    run = run_eval(
        _cases(3),
        {"baseline": explodes_once},
        {"exact": exact_match("answer")},
        progress=False,
    )

    assert len(run.rows) == 3, "the run kept going"
    broken = run.rows[1]
    assert broken.error is not None and "connection reset" in broken.error
    assert broken.scores == {}, "an error row has nothing to grade"
    assert not broken.passed


def test_a_task_that_reports_an_error_is_billed_but_not_graded() -> None:
    def refuses(case: Case) -> Outcome:
        del case
        return Outcome(usage=COSTLY, model=MODEL, error="stop_reason=max_tokens")

    run = run_eval(
        _cases(1),
        {"baseline": refuses},
        {"exact": exact_match("answer")},
        progress=False,
    )

    row = run.rows[0]
    assert row.error == "stop_reason=max_tokens"
    assert row.cost_usd == pytest.approx(0.03), "it still cost money, so it is still counted"
    assert run.spent_usd == pytest.approx(0.03)


def test_a_grader_that_spends_is_billed_to_the_same_ceiling() -> None:
    def paid_grader(case: Case, outcome: Outcome) -> Score:
        del case, outcome
        return Score(True, "judged", usage=COSTLY, model=MODEL)

    run = run_eval(
        _cases(2),
        {"baseline": _always("yes", COSTLY)},
        {"judge": paid_grader},
        progress=False,
    )

    # Two rows, each paying twice: once for the answer and once for the verdict.
    assert run.spent_usd == pytest.approx(0.12)
    assert run.rows[0].cost_usd == pytest.approx(0.06)


def test_a_grader_that_raises_fails_that_row_only() -> None:
    def broken_grader(case: Case, outcome: Outcome) -> Score:
        del case, outcome
        raise ValueError("rubric is empty")

    graders: dict[str, Grader] = {"exact": exact_match("answer"), "broken": broken_grader}
    run = run_eval(
        _cases(2),
        {"baseline": _always("yes")},
        graders,
        progress=False,
    )

    assert all(row.scores["exact"].passed for row in run.rows)
    assert all("rubric is empty" in row.scores["broken"].detail for row in run.rows)
    assert not run.rows[0].passed


def test_the_ceiling_stops_the_run_and_keeps_the_rows_it_paid_for() -> None:
    run = run_eval(
        _cases(5),
        {"baseline": _always("yes", COSTLY)},
        {"exact": exact_match("answer")},
        limit_usd=0.05,
        progress=False,
    )

    assert len(run.rows) == 2, "row 2 crossed $0.05; rows 3 to 5 never ran"
    assert all(row.passed for row in run.rows), "what finished is still usable"
    assert run.spent_usd == pytest.approx(0.06), "the crossing row was billed"
    assert run.stopped_early is not None
    assert "ENTROPIC_MAX_USD_PER_EVAL" in run.stopped_early, "name the right knob"


def test_a_run_needs_something_to_run_and_something_to_grade() -> None:
    with pytest.raises(ValueError, match="variant"):
        run_eval(_cases(1), {}, {"exact": exact_match("answer")})
    with pytest.raises(ValueError, match="grader"):
        run_eval(_cases(1), {"baseline": _always("yes")}, {})
