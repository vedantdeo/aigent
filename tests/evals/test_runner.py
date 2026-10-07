"""The runner's three promises: a bad row is data, a tripped ceiling keeps what it paid for, and
every dollar spent lands in the total.

None of these tests touch the network. The task under test is a plain function, which is the whole
reason the harness takes one instead of a prompt.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace

import pytest

from aigent.config import JUDGE_MODEL
from aigent.errors import BatchUnfinished, BudgetExceeded
from aigent.evals.dataset import Case
from aigent.evals.grade import Grader, Outcome, Score, exact_match
from aigent.evals.judge import Verdict
from aigent.evals.report import to_markdown
from aigent.evals.runner import EvalRun, Task, combine, run_eval
from aigent.llm.adapters.anthropic import usage_of
from aigent.llm.messages import Usage
from aigent.llm.pricing import usage_cost

from ..conftest import MakeJudge, Recorder

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


def test_a_grader_that_names_no_model_is_billed_at_the_runs_model() -> None:
    """A `Score` may carry usage without naming a model; the run's model is the only price left."""

    def paid_grader(case: Case, outcome: Outcome) -> Score:
        del case, outcome
        return Score(True, "judged", usage=COSTLY)

    run = run_eval(
        _cases(2),
        {"baseline": _always("yes", COSTLY)},
        {"judge": paid_grader},
        model=MODEL,
        progress=False,
    )

    # Two rows, each paying twice: once for the answer and once for the verdict, both on Opus.
    assert run.rows[0].cost_usd == pytest.approx(0.06)
    assert run.spent_usd == pytest.approx(0.12)


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
    assert "AIGENT_MAX_USD_PER_EVAL" in run.stopped_early, "name the right knob"


def test_progress_names_an_error_row_and_why_the_run_stopped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def costly_then_broken(case: Case) -> Outcome:
        if case.id == "c1":
            return Outcome(usage=COSTLY, model=MODEL, error="stop_reason=refusal")
        return Outcome(output={"answer": "yes"}, usage=COSTLY, model=MODEL)

    run_eval(_cases(3), {"a": costly_then_broken}, {"exact": exact_match("answer")}, limit_usd=0.05)

    out = capsys.readouterr().out
    assert "[1/3] c1 [a] ERROR stop_reason=refusal" in out
    assert "[2/3] c2 [a] exact=pass" in out
    assert "stopped early:" in out and "AIGENT_MAX_USD_PER_EVAL" in out


def test_a_run_needs_something_to_run_and_something_to_grade() -> None:
    with pytest.raises(ValueError, match="variant"):
        run_eval(_cases(1), {}, {"exact": exact_match("answer")})
    with pytest.raises(ValueError, match="grader"):
        run_eval(_cases(1), {"baseline": _always("yes")}, {})


# --- the judge inside the runner ----------------------------------------------------------------
# The judge's own tests call it directly. These two are the junction, which is where a signature or
# a billing mismatch would hide.


def test_an_llm_judge_is_just_another_grader_to_the_runner(make_judge: MakeJudge) -> None:
    judge, log = make_judge(Verdict(reasoning="names the direction", passed=True))
    answering = Usage(input_tokens=300, output_tokens=80)

    run = run_eval(
        _cases(2),
        {"baseline": _always("yes", answering)},
        {"exact": exact_match("answer"), "judge": judge},
        model=MODEL,
        progress=False,
    )

    assert len(log.prompts) == 2, "one verdict per row"
    assert all(row.scores["judge"].passed for row in run.rows)
    assert all(row.scores["judge"].model == JUDGE_MODEL for row in run.rows)

    # Each row pays twice, and the verdict is priced as the judge's model. Billing it at the task's
    # would be a quiet overcharge that still looked like a plausible total.
    per_row = usage_cost(MODEL, answering) + usage_cost(JUDGE_MODEL, usage_of(log.usage))
    assert run.rows[0].cost_usd == pytest.approx(per_row)
    assert run.spent_usd == pytest.approx(2 * per_row)


def test_a_judge_that_fails_on_the_network_costs_the_row_not_the_run(make_judge: MakeJudge) -> None:
    judge, _ = make_judge(fails_with=RuntimeError("503 upstream connect error"))

    run = run_eval(
        _cases(3),
        {"baseline": _always("yes")},
        {"exact": exact_match("answer"), "judge": judge},
        progress=False,
    )

    assert len(run.rows) == 3, "the run finished"
    assert all(row.scores["exact"].passed for row in run.rows), "the free grader still graded"
    assert all("503 upstream" in row.scores["judge"].detail for row in run.rows)


# --- combine ---------------------------------------------------------------------------------
# One run per variant, merged afterwards, because a retrieval label resolves to different chunk
# ids under each chunking strategy and so cannot be scored in one pass.


def _run(variant: str, *, digest: str = "abc123", grader: str = "exact") -> EvalRun:
    return run_eval(
        _cases(2),
        {variant: _always("yes")},
        {grader: exact_match("answer")},
        dataset="toy.jsonl",
        digest=digest,
        progress=False,
    )


def test_combine_reads_several_single_variant_runs_as_one_table() -> None:
    runs = [_run("fixed"), _run("by_sentence")]

    merged = combine(runs)

    assert merged.variants == ("fixed", "by_sentence"), "column order follows the runs given"
    assert len(merged.rows) == 4
    assert merged.graders == ("exact",)
    assert merged.started_at == min(run.started_at for run in runs), "the run began at the earliest"


@pytest.mark.parametrize(
    ("second", "says"),
    [
        pytest.param(
            lambda: _run("overlap", digest="different"),
            "different datasets",
            id="labels moved between the two runs",
        ),
        pytest.param(
            lambda: _run("overlap", grader="loose"),
            "different graders",
            id="columns would not be answering the same question",
        ),
        pytest.param(lambda: _run("fixed"), "share a variant name", id="one column hides another"),
    ],
)
def test_combine_refuses_runs_whose_columns_would_not_be_comparable(
    second: Callable[[], EvalRun], says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        combine([_run("fixed"), second()])


def test_combine_needs_something_to_combine() -> None:
    with pytest.raises(ValueError, match="at least one"):
        combine([])


@pytest.mark.parametrize(
    ("worst", "expectation", "ran"),
    [
        pytest.param(1.50, nullcontext(), 2, id="a run whose worst case fits goes ahead"),
        pytest.param(
            2.50,
            pytest.raises(BudgetExceeded, match="AIGENT_MAX_USD_PER_EVAL"),
            0,
            id="a run whose worst case does not is refused before its first row",
        ),
    ],
)
def test_a_run_is_admitted_whole_before_it_spends(
    worst: float, expectation: AbstractContextManager[object], ran: int
) -> None:
    called: list[str] = []

    def task(case: Case) -> Outcome:
        called.append(case.id)
        return Outcome(output={"answer": "yes"}, usage=COSTLY, model=MODEL)

    with expectation:
        run_eval(
            _cases(2),
            {"baseline": task},
            {"exact": exact_match("answer")},
            limit_usd=2.00,
            worst_usd=worst,
            progress=False,
        )
    assert len(called) == ran


# --- rows in flight at once ----------------------------------------------------------------------


def test_concurrent_rows_overlap_and_come_back_in_case_order() -> None:
    together = threading.Barrier(4, timeout=5)  # breaks unless four rows are in flight at once

    def task(case: Case) -> Outcome:
        together.wait()
        time.sleep((9 - int(case.id[1:])) / 1000)  # early cases finish last
        return Outcome(output={"answer": "yes"}, usage=COSTLY, model=MODEL)

    run = run_eval(
        _cases(8),
        {"a": task, "b": task},
        {"exact": exact_match("answer")},
        worst_usd=1.00,
        workers=4,
        progress=False,
    )

    assert [(row.case_id, row.variant) for row in run.rows] == [
        (f"c{i}", variant) for i in range(1, 9) for variant in ("a", "b")
    ]
    assert all(row.passed for row in run.rows)
    assert run.spent_usd == pytest.approx(16 * 0.03), "every row billed to the one budget"


def test_concurrent_rows_need_the_run_admitted_whole() -> None:
    called: list[str] = []

    def task(case: Case) -> Outcome:
        called.append(case.id)
        return Outcome(output={"answer": "yes"})

    with pytest.raises(ValueError, match="worst_usd"):
        run_eval(_cases(2), {"a": task}, {"exact": exact_match("answer")}, workers=2)
    assert called == []


def test_a_ceiling_tripped_mid_run_cancels_the_rows_not_yet_started() -> None:
    # A worst case that understates the run, the one way a concurrent run can still trip.
    run = run_eval(
        _cases(12),
        {"baseline": _always("yes", COSTLY)},
        {"exact": exact_match("answer")},
        limit_usd=0.05,
        worst_usd=0.01,
        workers=2,
        progress=False,
    )

    ran = [row.case_id for row in run.rows]
    assert 2 <= len(ran) < 12, ran
    assert ran == [f"c{i}" for i in range(1, len(ran) + 1)], "what ran is a prefix, in case order"
    assert run.spent_usd == pytest.approx(0.03 * len(ran)), "rows in flight were billed"
    assert run.stopped_early is not None and "AIGENT_MAX_USD_PER_EVAL" in run.stopped_early


# --- batch graders -------------------------------------------------------------------------------

BATCH_USAGE = Usage(input_tokens=1000, output_tokens=1000, batched=True)


class _BatchJudge:
    """A batch grader that passes a 'yes', bills `BATCH_USAGE` a verdict, and keeps each batch."""

    def __init__(self, fails_with: Exception | None = None) -> None:
        self.fails_with = fails_with
        self.batches: list[list[str]] = []
        self.called = 0

    def __call__(self, case: Case, outcome: Outcome) -> Score:
        self.called += 1
        return self._score(outcome)

    def grade_all(self, pairs: Sequence[tuple[Case, Outcome]]) -> list[Score]:
        if self.fails_with is not None:
            raise self.fails_with
        self.batches.append([case.id for case, _ in pairs])
        return [self._score(outcome) for _, outcome in pairs]

    def _score(self, outcome: Outcome) -> Score:
        passed = outcome.output.get("answer") == "yes"
        return Score(passed, "judged", usage=BATCH_USAGE, model=JUDGE_MODEL)


def test_a_batch_grader_grades_every_row_in_one_batch_after_they_run() -> None:
    judge = _BatchJudge()

    run = run_eval(
        _cases(2),
        {"a": _always("yes"), "b": _always("no")},
        {"judge": judge, "exact": exact_match("answer")},
        worst_usd=1.00,
        batch=True,
        progress=False,
    )

    assert judge.batches == [["c1", "c1", "c2", "c2"]] and judge.called == 0
    assert [list(row.scores) for row in run.rows] == [["judge", "exact"]] * 4, "graders' order"
    assert [row.passed for row in run.rows] == [True, False, True, False]
    each = usage_cost(JUDGE_MODEL, BATCH_USAGE)
    assert [row.cost_usd for row in run.rows] == pytest.approx([each] * 4)
    assert run.spent_usd == pytest.approx(4 * each), "billed at the batch price, to the one budget"
    assert run.batched == ("judge",)


@pytest.mark.parametrize(
    ("graders", "worst_usd", "says"),
    [
        pytest.param({"judge": _BatchJudge()}, None, "worst_usd", id="a run not admitted whole"),
        pytest.param(
            {"exact": exact_match("answer")}, 1.00, "can batch", id="no grader that can batch"
        ),
    ],
)
def test_batch_grading_is_refused_before_any_task_runs(
    graders: dict[str, Grader], worst_usd: float | None, says: str
) -> None:
    called: list[str] = []

    def task(case: Case) -> Outcome:
        called.append(case.id)
        return Outcome(output={"answer": "yes"})

    with pytest.raises(ValueError, match=says):
        run_eval(_cases(2), {"a": task}, graders, worst_usd=worst_usd, batch=True)
    assert called == []


def test_without_batch_a_batch_grader_grades_each_row_as_it_runs() -> None:
    judge = _BatchJudge()

    run_eval(_cases(2), {"a": _always("yes")}, {"judge": judge}, worst_usd=1.00, progress=False)

    assert (judge.called, judge.batches) == (2, [])


def test_an_error_row_is_not_sent_to_the_batch() -> None:
    judge = _BatchJudge()

    def broken_on_c2(case: Case) -> Outcome:
        if case.id == "c2":
            return Outcome(error="stop_reason=refusal")
        return Outcome(output={"answer": "yes"})

    run = run_eval(
        _cases(3), {"a": broken_on_c2}, {"judge": judge}, worst_usd=1.00, batch=True, progress=False
    )

    assert judge.batches == [["c1", "c3"]]
    assert run.rows[1].error == "stop_reason=refusal" and run.rows[1].scores == {}


def test_a_run_that_tripped_sends_no_batch_and_fails_what_it_could_not_grade() -> None:
    judge = _BatchJudge()

    run = run_eval(
        _cases(5),
        {"a": _always("yes", COSTLY)},
        {"judge": judge},
        limit_usd=0.05,
        worst_usd=0.01,
        batch=True,
        progress=False,
    )

    assert judge.batches == []
    assert run.stopped_early is not None
    assert all(not row.passed for row in run.rows)
    assert all("not graded" in row.scores["judge"].detail for row in run.rows)


@pytest.mark.parametrize(
    ("lost", "stopped"),
    [
        pytest.param(
            BatchUnfinished("batch msgbatch_7 was not collected"),
            "msgbatch_7",
            id="a batch that may still bill is named as why the run stopped",
        ),
        pytest.param(ConnectionError("reset"), None, id="a batch never sent fails only its rows"),
    ],
)
def test_a_lost_batch_fails_its_rows_and_not_the_run(lost: Exception, stopped: str | None) -> None:
    run = run_eval(
        _cases(2),
        {"a": _always("yes")},
        {"judge": _BatchJudge(fails_with=lost), "exact": exact_match("answer")},
        worst_usd=1.00,
        batch=True,
        progress=False,
    )

    assert [row.scores["exact"].passed for row in run.rows] == [True, True]
    assert all(f"grader raised {type(lost).__name__}" in r.scores["judge"].detail for r in run.rows)
    assert (run.stopped_early is None) if stopped is None else stopped in str(run.stopped_early)


def test_each_row_trace_carries_its_batch_verdict(traces: Recorder) -> None:
    run_eval(
        _cases(2),
        {"a": _always("yes"), "b": _always("no")},
        {"judge": _BatchJudge()},
        dataset="toy",
        worst_usd=1.00,
        batch=True,
        progress=False,
    )

    rows = [node for node in traces.roots if node.name == "toy"]
    assert [node.scores["judge"][0] for node in rows] == [True, False, True, False]
    [batch] = [node for node in traces.every() if node.name == "judge batch"]
    assert batch.kind == "evaluator" and batch.output == {"rows": 4, "passed": 2}


def test_concurrent_rows_are_batch_graded_in_case_order() -> None:
    judge = _BatchJudge()

    run = run_eval(
        _cases(6),
        {"a": _always("yes")},
        {"judge": judge},
        worst_usd=1.00,
        workers=3,
        batch=True,
        progress=False,
    )

    assert judge.batches == [[f"c{i}" for i in range(1, 7)]]
    assert all(row.passed for row in run.rows)


def test_the_report_says_which_graders_ran_as_a_batch() -> None:
    run = run_eval(
        _cases(1), {"a": _always("yes")}, {"judge": _BatchJudge()}, worst_usd=1.00, batch=True
    )

    assert "batch-graded: `judge`" in to_markdown(run)
    assert "batch-graded" not in to_markdown(replace(run, batched=()))
