"""The report is the deliverable, so its numbers are worth a test of their own."""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue

from aigent.config import MAX_USD_PER_EVAL, MODEL
from aigent.evals.dataset import Case
from aigent.evals.grade import (
    Outcome,
    Score,
    exact_match,
    field_match,
    recall_at_k,
    reciprocal_rank,
)
from aigent.evals.report import to_markdown, write_report
from aigent.evals.runner import EvalRun, RowResult, Task, run_eval
from aigent.messages import Usage

FIELDS = ("company", "quarter")


def _cases(n: int) -> list[Case]:
    return [
        Case.model_validate(
            {
                "id": f"c{i}",
                "input": {"headline": f"headline {i}"},
                "expected": {"answer": "yes", "company": "Infosys", "quarter": "Q3"},
            }
        )
        for i in range(1, n + 1)
    ]


def _task(answer: str, quarter: str, usage: Usage | None = None) -> Task:
    def task(case: Case) -> Outcome:
        del case
        return Outcome(
            output={"answer": answer, "company": "Infosys", "quarter": quarter},
            usage=usage,
            model=MODEL,
        )

    return task


def _run(limit_usd: float = MAX_USD_PER_EVAL) -> EvalRun:
    return run_eval(
        _cases(4),
        {"baseline": _task("no", "Q2"), "better": _task("yes", "Q3")},
        {"exact": exact_match("answer"), "fields": field_match(FIELDS)},
        dataset="headlines.jsonl",
        digest="deadbeef1234",
        limit_usd=limit_usd,
        progress=False,
    )


def test_summary_table_has_one_row_per_variant_with_its_pass_rate() -> None:
    report = to_markdown(_run())

    assert "| variant | exact | fields | errors | cost |" in report
    assert "| baseline | 0/4 (0%) | 0/4 (0%) | 0 | $0.00000 |" in report
    assert "| better | 4/4 (100%) | 4/4 (100%) | 0 | $0.00000 |" in report


def test_per_field_table_separates_the_field_that_broke() -> None:
    report = to_markdown(_run())

    # company is right under both variants; only quarter distinguishes them.
    assert "| company | 4/4 (100%) | 4/4 (100%) |" in report
    assert "| quarter | 0/4 (0%) | 4/4 (100%) |" in report


def test_header_records_what_makes_two_runs_comparable() -> None:
    report = to_markdown(_run())

    assert "`headlines.jsonl` (`deadbeef1234`, 4 cases)" in report
    assert f"`{MODEL}`" in report


def test_failures_are_listed_with_the_reason_and_capped() -> None:
    report = to_markdown(_run(), max_failures=2)

    assert "## Failures" in report
    assert "`c1` [baseline]" in report
    assert "expected 'yes', got 'no'" in report
    assert "not shown" in report, "a long failure list is truncated, not dumped"


def test_error_rows_are_counted_apart_from_wrong_answers() -> None:
    run = EvalRun(
        dataset="toy.jsonl",
        digest="abc",
        model=MODEL,
        variants=("baseline",),
        graders=("exact",),
        rows=[
            RowResult("c1", "baseline", scores={"exact": Score(True)}),
            RowResult("c2", "baseline", scores={}, error="429 rate limited"),
        ],
    )
    report = to_markdown(run)

    # One graded row, one error. The pass rate is 1/1, not 1/2: a rate limit is not a wrong answer.
    assert "| baseline | 1/1 (100%) | 1 | $0.00000 |" in report
    assert "error: 429 rate limited" in report


def test_a_partial_run_says_so_at_the_top() -> None:
    """A truncated run is still worth reading, but never worth mistaking for a whole one."""
    run = run_eval(
        _cases(5),
        {"baseline": _task("yes", "Q3", Usage(input_tokens=1000, output_tokens=1000))},
        {"exact": exact_match("answer")},
        dataset="headlines.jsonl",
        limit_usd=0.05,
        progress=False,
    )
    report = to_markdown(run)

    assert run.stopped_early is not None
    assert "**partial run" in report
    assert "2 cases" in report, "the header counts what ran, not what was asked for"


def test_write_report_names_the_file_after_the_dataset_and_the_run(tmp_path: Path) -> None:
    run = _run()
    path = write_report(run, tmp_path)

    assert path.parent == tmp_path
    assert path.name.startswith("headlines-")
    assert path.suffix == ".md"
    assert path.read_text(encoding="utf-8") == to_markdown(run)


def test_the_header_names_a_grading_model_only_when_a_grader_spent() -> None:
    """Two models, two roles. A report that names only one is not reproducible.

    The line is derived from what the rows actually cost, not declared, so both halves of that
    branch are worth pinning: it appears when a grader spent, and stays away when none did.
    """

    def judge(case: Case, outcome: Outcome) -> Score:
        del case, outcome
        return Score(
            True, "fine", usage=Usage(input_tokens=100, output_tokens=20), model="claude-sonnet-5"
        )

    run = run_eval(
        _cases(2),
        {"baseline": _task("yes", "Q3", Usage(input_tokens=300, output_tokens=80))},
        {"judge": judge},
        dataset="headlines.jsonl",
        model=MODEL,
        progress=False,
    )
    report = to_markdown(run)

    assert f"- model: `{MODEL}`" in report
    assert "- graded by: `claude-sonnet-5`" in report
    assert "graded by" not in to_markdown(_run()), "free graders name no model"


# --- The metrics table -----------------------------------------------------------------------
# A grader whose verdict is a number needs a different row in the report from one whose verdict is
# a verdict, so these check both that the mean appears and that it stays out of an extraction run.


def _retrieval_run() -> EvalRun:
    cases = [
        Case.model_validate(
            {"id": f"q{i}", "input": {"question": "?"}, "expected": {"relevant": ["b"]}}
        )
        for i in range(1, 5)
    ]

    def retriever(ranked: list[JsonValue]) -> Task:
        def task(case: Case) -> Outcome:
            del case
            return Outcome(output={"retrieved": ranked})

        return task

    return run_eval(
        cases,
        {"strong": retriever(["b", "x"]), "weak": retriever(["x", "b"])},
        {"recall@1": recall_at_k(1), "mrr": reciprocal_rank()},
        dataset="questions.jsonl",
        digest="c0ffee123456",
        progress=False,
    )


def test_the_metrics_table_means_each_numeric_grader_per_variant() -> None:
    report = to_markdown(_retrieval_run())

    assert "## Metrics (mean per case)" in report
    assert "| metric | strong | weak |" in report
    assert "| recall@1 | 1.000 | 0.000 |" in report
    assert "| mrr | 1.000 | 0.500 |" in report


def test_the_pass_rate_and_the_mean_answer_different_questions_about_the_same_grader() -> None:
    """`weak` finds the right chunk every time and never at rank 1. Both facts belong here."""
    report = to_markdown(_retrieval_run())

    assert "| weak | 0/4 (0%) | 0/4 (0%) | 0 | $0.00000 |" in report
    assert "| mrr | 1.000 | 0.500 |" in report


def test_a_run_whose_graders_report_no_numbers_prints_no_metrics_table() -> None:
    assert "## Metrics" not in to_markdown(_run())
