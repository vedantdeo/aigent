"""The runner: iterate cases, run each variant, grade, and keep the money in bounds.

A task that raises becomes an error row rather than ending the run; a tripped ceiling stops the run
and keeps the rows already paid for; every dollar is billed to one budget. Cases are the outer loop
so a truncated run is still a comparison. Rows may run concurrently, and batch graders may grade
every row in one batch, once the whole run is admitted.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime

from aigent import tracing
from aigent.config import EVAL_WORKERS, MAX_USD_PER_EVAL, MODEL
from aigent.errors import BatchUnfinished
from aigent.evals.dataset import Case
from aigent.evals.grade import BatchGrader, Grader, Outcome, Score
from aigent.llm.messages import Usage
from aigent.llm.pricing import Budget

Task = Callable[[Case], Outcome]
Job = tuple[Case, str, Task]


@dataclass(frozen=True)
class _Ran:
    """What a row left for its batch graders: the case, what the task returned, and its trace."""

    case: Case
    outcome: Outcome
    seen: tracing.Observation


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
    batched: tuple[str, ...] = ()  # graders that graded every row in one batch

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
    workers: int = EVAL_WORKERS,
    batch: bool = False,
) -> EvalRun:
    """Run every case through every variant, grade each outcome, return the whole run.

    `model` is recorded, not applied — the task owns its own call. Pass the one it really uses.
    `worst_usd` is the whole run's worst case; given, the run is refused before its first row
    unless all of it fits under `limit_usd`. `workers` rows run at once, which needs `worst_usd`;
    rows come back in case order however they finish. With `batch`, every `BatchGrader` grades all
    the rows in one batch once they have run, which also needs `worst_usd`.
    """
    if not variants:
        raise ValueError("run_eval needs at least one variant")
    if not graders:
        raise ValueError("run_eval needs at least one grader")
    if workers > 1 and worst_usd is None:
        raise ValueError(
            "concurrent rows need worst_usd: rows in flight cannot be recalled, so only a run "
            "admitted whole before it starts is sure to stay under its ceiling"
        )
    batched = {n: g for n, g in graders.items() if batch and isinstance(g, BatchGrader)}
    if batch and not batched:
        raise ValueError("batch grading needs a grader that can batch, such as an LlmJudge")
    if batch and worst_usd is None:
        raise ValueError(
            "batch grading needs worst_usd: a submitted batch cannot be recalled, so only a run "
            "admitted whole before it starts is sure to stay under its ceiling"
        )
    inline = {name: grader for name, grader in graders.items() if name not in batched}

    run = EvalRun(
        dataset=dataset,
        digest=digest,
        model=model,
        variants=tuple(variants),
        graders=tuple(graders),
        limit_usd=limit_usd,
        batched=tuple(batched),
    )
    budget = Budget(limit_usd=limit_usd, scope="eval")
    if worst_usd is not None:
        budget.admit(worst_usd)

    jobs: list[Job] = [(case, name, task) for case in cases for name, task in variants.items()]
    session = f"{dataset}-{run.started_at:%Y%m%d-%H%M%S}"
    tracer = tracing.tracer()  # chosen here, before any row could race to choose it
    done = 0
    ran: dict[tuple[str, str], _Ran] = {}  # one item set per row, atomic without a lock

    def traced(case: Case, name: str, task: Task) -> RowResult:
        with tracer.observe(
            dataset,
            inputs=case.input,
            metadata={"case": case.id, "variant": name, "expected": case.expected},
            tags=(dataset, name, model, *case.tags),
            session=session,
        ) as seen:
            row, outcome = _run_one(case, name, task, inline, budget, model, seen)
        if batched and outcome is not None and row.error is None:
            ran[case.id, name] = _Ran(case, outcome, seen)
        return row

    def finished(row: RowResult) -> None:
        nonlocal done
        done += 1
        if progress:
            print(f"[{done}/{len(jobs)}] {_describe(row)}")

    if workers > 1:
        run.rows = _concurrently(jobs, traced, finished, budget, workers)
    else:
        for job in jobs:
            run.rows.append(row := traced(*job))
            finished(row)
            if budget.tripped is not None:
                break

    run.stopped_early = budget.tripped
    if batched:
        run.rows, unfinished = _grade_in_batches(run.rows, ran, batched, graders, budget, model)
        run.stopped_early = run.stopped_early or unfinished
        if progress:
            print(f"batch-graded {len(ran)} rows: {sum(row.passed for row in run.rows)} passed")
    run.spent_usd = budget.spent_usd
    tracer.flush()
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
        batched=tuple(dict.fromkeys(name for run in runs for name in run.batched)),
    )


def _concurrently(
    jobs: Sequence[Job],
    traced: Callable[[Case, str, Task], RowResult],
    finished: Callable[[RowResult], None],
    budget: Budget,
    workers: int,
) -> list[RowResult]:
    """Every job on a pool of `workers`, in job order; a tripped ceiling cancels what has not
    started, and keeps the rows in flight, which are already paid for."""
    rows: dict[int, RowResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Each row runs in a copy of this context, so its trace nests where the run was started.
        sent: dict[Future[RowResult], int] = {
            pool.submit(contextvars.copy_context().run, traced, *job): index
            for index, job in enumerate(jobs)
        }
        for future in as_completed(sent):
            if future.cancelled():
                continue
            rows[sent[future]] = row = future.result()
            finished(row)
            if budget.tripped is not None:
                for waiting in sent:
                    waiting.cancel()
    return [rows[index] for index in sorted(rows)]


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
    seen: tracing.Observation,
) -> tuple[RowResult, Outcome | None]:
    """One case, one variant: run it, bill it, grade it, under the row's trace. Never raises."""
    try:
        outcome = task(case)
    except Exception as exc:  # noqa: BLE001 - a broken task is a row, not the end of the run
        error = f"{type(exc).__name__}: {exc}"
        seen.finish(error=error)
        return RowResult(case.id, variant, scores={}, error=error), None

    cost = _bill(budget, outcome.usage, outcome.model or model)
    if outcome.error is not None:
        seen.finish(outcome.output, error=outcome.error, metadata={"cost_usd": cost})
        return RowResult(case.id, variant, scores={}, cost_usd=cost, error=outcome.error), outcome

    scores: dict[str, Score] = {}
    for grader_name, grader in graders.items():
        with tracing.tracer().observe(grader_name, "evaluator") as graded:
            try:
                score = grader(case, outcome)
            except Exception as exc:  # noqa: BLE001 - the judge can fail on the network like anything
                score = Score(False, f"grader raised {type(exc).__name__}: {exc}")
            graded.finish({"passed": score.passed, "detail": score.detail})
        scores[grader_name] = score
        seen.score(grader_name, score.passed, score.detail)
        cost += _bill(budget, score.usage, score.model or model)

    seen.finish(outcome.output, metadata={"cost_usd": cost})
    return RowResult(case.id, variant, scores=scores, cost_usd=cost), outcome


def _grade_in_batches(
    rows: Sequence[RowResult],
    ran: Mapping[tuple[str, str], _Ran],
    batched: Mapping[str, BatchGrader],
    graders: Mapping[str, Grader],
    budget: Budget,
    model: str,
) -> tuple[list[RowResult], str | None]:
    """Each batch grader over every row that ran, then each row's scores in the graders' order.
    Returns the rows, and why a batch went uncollected if one did."""
    keys = [(row.case_id, row.variant) for row in rows if (row.case_id, row.variant) in ran]
    pairs = [(ran[key].case, ran[key].outcome) for key in keys]
    added: dict[tuple[str, str], dict[str, Score]] = {key: {} for key in keys}
    costs = dict.fromkeys(keys, 0.0)
    unfinished: str | None = None
    for name, grader in batched.items():
        if budget.tripped is not None:
            scores = [Score(False, "not graded: the run stopped before its batch was sent")] * len(
                pairs
            )
        else:
            with tracing.tracer().observe(f"{name} batch", "evaluator") as graded:
                try:
                    scores = grader.grade_all(pairs) if pairs else []
                except Exception as exc:  # noqa: BLE001 - a lost batch fails its rows, not the run
                    if isinstance(exc, BatchUnfinished):
                        unfinished = str(exc)
                    scores = [Score(False, f"grader raised {type(exc).__name__}: {exc}")] * len(
                        pairs
                    )
                graded.finish({"rows": len(scores), "passed": sum(s.passed for s in scores)})
        for key, score in zip(keys, scores, strict=True):
            added[key][name] = score
            ran[key].seen.score(name, score.passed, score.detail)
            costs[key] += _bill(budget, score.usage, score.model or model)
    graded_rows = [
        row
        if (row.case_id, row.variant) not in added
        else _with(row, added[row.case_id, row.variant], costs[row.case_id, row.variant], graders)
        for row in rows
    ]
    return graded_rows, unfinished


def _with(
    row: RowResult, added: Mapping[str, Score], cost: float, graders: Mapping[str, Grader]
) -> RowResult:
    """`row` with its batch scores added, every score in the graders' order, and their cost."""
    scores = {**row.scores, **added}
    ordered = {name: scores[name] for name in graders if name in scores}
    return RowResult(row.case_id, row.variant, ordered, row.cost_usd + cost, row.error)


def _describe(row: RowResult) -> str:
    if row.error is not None:
        return f"{row.case_id} [{row.variant}] ERROR {row.error}"
    verdicts = " ".join(
        f"{name}={'pass' if score.passed else 'FAIL'}" for name, score in row.scores.items()
    )
    return f"{row.case_id} [{row.variant}] {verdicts}  ${row.cost_usd:.5f}"
