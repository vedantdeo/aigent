"""The runner: iterate cases, run each variant, grade, and keep the money in bounds.

A task that raises becomes an error row rather than ending the run; a tripped ceiling stops the run
and keeps the rows already paid for; every dollar is billed to one budget. Cases are the outer loop
so a truncated run is still a comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from aigent.config import MAX_USD_PER_EVAL, MODEL
from aigent.evals.dataset import Case
from aigent.evals.grade import Grader, Outcome, Score
from aigent.messages import Usage
from aigent.pricing import Budget

Task = Callable[[Case], Outcome]


@dataclass(frozen=True)
class RowResult:
    """One case under one variant: what every grader said, and what it cost."""

    case_id: str
    variant: str
    scores: dict[str, Score]
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Every grader passed. An error row never passes, whatever the graders managed to say."""
        return (
            self.error is None and bool(self.scores) and all(s.passed for s in self.scores.values())
        )


@dataclass
class EvalRun:
    """Everything needed to reproduce and compare one run.

    Model, dataset digest and date are what tell you whether two scores are comparable.
    """

    dataset: str
    digest: str
    model: str
    variants: tuple[str, ...]
    graders: tuple[str, ...]
    rows: list[RowResult] = field(default_factory=list)
    spent_usd: float = 0.0
    limit_usd: float = MAX_USD_PER_EVAL
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    stopped_early: str | None = None

    @property
    def n_cases(self) -> int:
        return len({row.case_id for row in self.rows})

    def rows_for(self, variant: str) -> list[RowResult]:
        return [row for row in self.rows if row.variant == variant]


def run_eval(
    cases: Sequence[Case],
    variants: Mapping[str, Task],
    graders: Mapping[str, Grader],
    *,
    dataset: str = "unnamed",
    digest: str = "",
    model: str = MODEL,
    limit_usd: float = MAX_USD_PER_EVAL,
    worst_usd: float | None = None,
    progress: bool = True,
) -> EvalRun:
    """Run every case through every variant, grade each outcome, return the whole run.

    `model` is recorded, not applied — the task owns its own call. Pass the one it really uses.
    `worst_usd` is the whole run's worst case; given, the run is refused before its first row
    unless all of it fits under `limit_usd`.
    """
    if not variants:
        raise ValueError("run_eval needs at least one variant")
    if not graders:
        raise ValueError("run_eval needs at least one grader")

    run = EvalRun(
        dataset=dataset,
        digest=digest,
        model=model,
        variants=tuple(variants),
        graders=tuple(graders),
        limit_usd=limit_usd,
    )
    budget = Budget(limit_usd=limit_usd, scope="eval")
    if worst_usd is not None:
        budget.admit(worst_usd)

    total = len(cases) * len(variants)
    done = 0

    for case in cases:
        for name, task in variants.items():
            done += 1
            row = _run_one(case, name, task, graders, budget, model)
            run.rows.append(row)
            if progress:
                print(f"[{done}/{total}] {_describe(row)}")
            if budget.tripped is not None:
                run.stopped_early = budget.tripped
                break
        if budget.tripped is not None:
            break

    run.spent_usd = budget.spent_usd
    if progress:
        print(f"\n{len(run.rows)} rows, ${run.spent_usd:.5f} of a ${limit_usd:.2f} ceiling")
        if run.stopped_early:
            print(f"stopped early: {run.stopped_early}")
    return run


def combine(runs: Sequence[EvalRun]) -> EvalRun:
    """Several single-variant runs read as one comparison table.

    Needed where variants disagree about the *label* and not only the task: a retrieval label is a
    quote, and it resolves to different chunk ids under each chunking strategy, so one `run_eval`
    call cannot serve them all. Refuses runs over different datasets or graders, and repeated
    variant names — a table of those columns would be comparing nothing.
    """
    if not runs:
        raise ValueError("combine needs at least one run")
    first = runs[0]
    for run in runs[1:]:
        if (run.dataset, run.digest) != (first.dataset, first.digest):
            raise ValueError(
                f"runs graded different datasets: {first.dataset}@{first.digest} "
                f"and {run.dataset}@{run.digest}"
            )
        if run.graders != first.graders:
            raise ValueError(f"runs used different graders: {first.graders} and {run.graders}")
    variants = tuple(variant for run in runs for variant in run.variants)
    if len(set(variants)) != len(variants):
        raise ValueError(f"two runs share a variant name, so one would hide the other: {variants}")

    return EvalRun(
        dataset=first.dataset,
        digest=first.digest,
        model=first.model,
        variants=variants,
        graders=first.graders,
        rows=[row for run in runs for row in run.rows],
        spent_usd=sum(run.spent_usd for run in runs),
        limit_usd=first.limit_usd,
        started_at=min(run.started_at for run in runs),
        stopped_early=next((run.stopped_early for run in runs if run.stopped_early), None),
    )


def _bill(budget: Budget, usage: Usage | None, model: str) -> float:
    """Charge a call that may not have happened; a task that never touched the API costs nothing."""
    return budget.charge(model, usage) if usage is not None else 0.0


def _run_one(
    case: Case,
    variant: str,
    task: Task,
    graders: Mapping[str, Grader],
    budget: Budget,
    model: str,
) -> RowResult:
    """One case, one variant: run it, bill it, grade it. Never raises."""
    try:
        outcome = task(case)
    except Exception as exc:  # noqa: BLE001 - a broken task is a row, not the end of the run
        return RowResult(case.id, variant, scores={}, error=f"{type(exc).__name__}: {exc}")

    cost = _bill(budget, outcome.usage, outcome.model or model)
    if outcome.error is not None:
        return RowResult(case.id, variant, scores={}, cost_usd=cost, error=outcome.error)

    scores: dict[str, Score] = {}
    for grader_name, grader in graders.items():
        try:
            score = grader(case, outcome)
        except Exception as exc:  # noqa: BLE001 - the judge can fail on the network like anything
            score = Score(False, f"grader raised {type(exc).__name__}: {exc}")
        scores[grader_name] = score
        cost += _bill(budget, score.usage, score.model or model)

    return RowResult(case.id, variant, scores=scores, cost_usd=cost)


def _describe(row: RowResult) -> str:
    if row.error is not None:
        return f"{row.case_id} [{row.variant}] ERROR {row.error}"
    verdicts = " ".join(
        f"{name}={'pass' if score.passed else 'FAIL'}" for name, score in row.scores.items()
    )
    return f"{row.case_id} [{row.variant}] {verdicts}  ${row.cost_usd:.5f}"
