"""The eval harness: run a task over a labelled dataset, grade it, print a table you can defend.

Built in Week 1 and reused by everything after it. The shape that makes that reuse work is the
split between the *task* and the *harness*:

    Case    one row of a JSONL dataset: id, input, expected, tags
    Task    Callable[[Case], Outcome] — owns its own API call, if it makes one at all
    Grader  Callable[[Case, Outcome], Score] — a verdict, and per-field detail when it has it

The harness never assumes the thing under test is a model call. That is deliberate: Week 2 grades a
retrieval function that spends nothing, and Week 4 grades an agent trajectory. Same runner.

    from aigent.evals import Case, load_jsonl, exact_match, run_eval, write_report
"""

from aigent.errors import DatasetError
from aigent.evals.dataset import Case, digest, load_jsonl
from aigent.evals.grade import (
    Grader,
    Outcome,
    Score,
    contains,
    exact_match,
    field_match,
    flag,
    hit_at_k,
    pydantic_valid,
    recall_at_k,
    reciprocal_rank,
    regex,
    resolvable,
)
from aigent.evals.judge import LlmJudge, Verdict
from aigent.evals.report import to_markdown, write_report
from aigent.evals.runner import EvalRun, RowResult, Task, combine, run_eval

__all__ = [
    "Case",
    "DatasetError",
    "EvalRun",
    "Grader",
    "LlmJudge",
    "Outcome",
    "RowResult",
    "Score",
    "Task",
    "Verdict",
    "combine",
    "contains",
    "digest",
    "exact_match",
    "field_match",
    "flag",
    "hit_at_k",
    "load_jsonl",
    "pydantic_valid",
    "recall_at_k",
    "reciprocal_rank",
    "regex",
    "resolvable",
    "run_eval",
    "to_markdown",
    "write_report",
]
