"""The runner: iterate cases, run each variant, grade, and keep the money in bounds.

Three behaviours matter more than the loop itself.

  - **A task that raises is an error row, not a crashed run.** Same reasoning as tool errors in the
    agent loop: a failure is data about the system under test, and losing the other forty rows to
    it helps nobody.
  - **The per-eval ceiling stops the run and keeps what is finished.** `Budget.add` raises, which is
    right for a chat session and wrong here, so the runner catches it, marks the run partial, and
    reports the rows it already paid for.
  - **Every dollar is attributed.** The task reports its usage in the `Outcome`; a grader that
    spends (the judge) reports its usage in the `Score`. Both are billed to the same budget, so the
    printed total is the real total.

Cases are the outer loop and variants the inner one, so a run that stops early has every variant
for the first k cases rather than all of variant A and none of variant B. A truncated comparison is
still a comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from anthropic.types import Usage

from entropic.config import MAX_USD_PER_EVAL, MODEL
from entropic.evals.dataset import Case
from entropic.evals.grade import Grader, Outcome, Score
from entropic.pricing import Budget

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

    The model, the dataset digest and the date are not decoration: a score is only meaningful next
    to another score, and these are what tell you whether the two are comparable.
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
    progress: bool = True,
) -> EvalRun:
    """Run every case through every variant, grade each outcome, return the whole run.

    `model` is recorded, not applied — the task owns its own call and therefore its own model. Pass
    the one the task actually uses, or the report will describe a run that never happened.
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


def _bill(budget: Budget, usage: Usage | None, model: str) -> float:
    """Charge a call that may not have happened.

    A task that never touched the API costs nothing, and that is a first-class case here: Week 2
    grades a retrieval function that returns `Outcome(usage=None)`.
    """
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
