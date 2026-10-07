"""The results table: what you paste into a README and compare against next week.

Four tables — summary, metrics, per field, failures. The middle two are omitted when no grader fed
them. Error rows are excluded from every denominator.
"""

from __future__ import annotations

from pathlib import Path

from aigent.config import MAX_FAILURES_SHOWN
from aigent.evals.runner import EvalRun, RowResult

REPORTS_DIR = Path(__file__).resolve().parents[3] / "evals" / "reports"


def _rate(passed: int, graded: int) -> str:
    if graded == 0:
        return "—"
    return f"{passed}/{graded} ({100 * passed / graded:.0f}%)"


def _graded(rows: list[RowResult]) -> list[RowResult]:
    return [row for row in rows if row.error is None]


def to_markdown(run: EvalRun, max_failures: int = MAX_FAILURES_SHOWN) -> str:
    """The whole report as one Markdown string."""
    lines: list[str] = [f"# {run.dataset}", ""]
    lines += _header_block(run)
    lines += ["", "## Results", ""]
    lines += _summary_table(run)

    metrics = _metrics_table(run)
    if metrics:
        lines += ["", "## Metrics (mean per case)", ""]
        lines += [
            "Each metric is meaned over the cases it could score, not over the whole dataset.",
            "",
        ]
        lines += metrics

    per_field = _per_field_table(run)
    if per_field:
        lines += ["", "## Per field", ""]
        lines += per_field

    failures = _failure_list(run, max_failures)
    if failures:
        lines += ["", "## Failures", ""]
        lines += failures

    return "\n".join(lines) + "\n"


def _header_block(run: EvalRun) -> list[str]:
    digest = run.digest or "unrecorded"
    block = [
        f"- dataset: `{run.dataset}` (`{digest}`, {run.n_cases} cases)",
        f"- model: `{run.model}`",
    ]
    graders = _grading_models(run)
    if graders:
        block.append(f"- graded by: {', '.join(f'`{m}`' for m in graders)}")
    if run.batched:
        names = ", ".join(f"`{name}`" for name in run.batched)
        block.append(f"- batch-graded: {names}, through the Message Batches API at half price")
    block += [
        f"- run: {run.started_at:%Y-%m-%d %H:%M} UTC",
        f"- cost: ${run.spent_usd:.5f} of a ${run.limit_usd:.2f} ceiling",
    ]
    if run.stopped_early:
        block.append(f"- **partial run — {run.stopped_early}**")
    return block


def _grading_models(run: EvalRun) -> list[str]:
    """Models a grader actually spent on, read back from the scores rather than from config."""
    seen: list[str] = []
    for row in run.rows:
        for score in row.scores.values():
            if score.model is not None and score.model not in seen:
                seen.append(score.model)
    return seen


def _summary_table(run: EvalRun) -> list[str]:
    header = "| variant | " + " | ".join(run.graders) + " | errors | cost |"
    rule = "|---" * (len(run.graders) + 3) + "|"
    rows = [header, rule]
    for variant in run.variants:
        all_rows = run.rows_for(variant)
        graded = _graded(all_rows)
        cells = [variant]
        for grader in run.graders:
            passed = sum(
                1 for row in graded if (s := row.scores.get(grader)) is not None and s.passed
            )
            cells.append(_rate(passed, len(graded)))
        cells.append(str(len(all_rows) - len(graded)))
        cells.append(f"${sum(row.cost_usd for row in all_rows):.5f}")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _metrics_table(run: EvalRun) -> list[str]:
    """One row per grader that reported a number, meaned over the cases it graded.

    Apart from the summary because it answers a different question: a retriever finding two of three
    relevant chunks every time reads 0% there and 0.667 here.
    """
    scored = [
        grader
        for grader in run.graders
        if any(
            (score := row.scores.get(grader)) is not None and score.value is not None
            for row in run.rows
        )
    ]
    if not scored:
        return []

    header = "| metric | " + " | ".join(run.variants) + " |"
    rule = "|---" * (len(run.variants) + 1) + "|"
    table = [header, rule]
    for grader in scored:
        cells = [grader]
        for variant in run.variants:
            values = [
                score.value
                for row in _graded(run.rows_for(variant))
                if (score := row.scores.get(grader)) is not None and score.value is not None
            ]
            cells.append(f"{sum(values) / len(values):.3f}" if values else "—")
        table.append("| " + " | ".join(cells) + " |")
    return table


def _per_field_table(run: EvalRun) -> list[str]:
    """One row per field, one column per variant. Empty when no grader reported field detail."""
    fields: list[str] = []
    for row in run.rows:
        for score in row.scores.values():
            for name in score.parts:
                if name not in fields:
                    fields.append(name)
    if not fields:
        return []

    header = "| field | " + " | ".join(run.variants) + " |"
    rule = "|---" * (len(run.variants) + 1) + "|"
    table = [header, rule]
    for name in fields:
        cells = [name]
        for variant in run.variants:
            passed = graded = 0
            for row in _graded(run.rows_for(variant)):
                for score in row.scores.values():
                    if name in score.parts:
                        graded += 1
                        passed += int(score.parts[name])
            cells.append(_rate(passed, graded))
        table.append("| " + " | ".join(cells) + " |")
    return table


def _failure_list(run: EvalRun, limit: int) -> list[str]:
    lines: list[str] = []
    shown = 0
    for row in run.rows:
        if row.passed:
            continue
        if shown >= limit:
            lines.append(f"- …and more; {_failure_count(run) - shown} not shown")
            break
        shown += 1
        if row.error is not None:
            lines.append(f"- `{row.case_id}` [{row.variant}] — error: {row.error}")
            continue
        why = "; ".join(
            f"**{name}**: {score.detail or 'failed'}"
            for name, score in row.scores.items()
            if not score.passed
        )
        lines.append(f"- `{row.case_id}` [{row.variant}] — {why}")
    return lines


def _failure_count(run: EvalRun) -> int:
    return sum(1 for row in run.rows if not row.passed)


def write_report(run: EvalRun, directory: Path = REPORTS_DIR) -> Path:
    """Write the report next to its siblings and return the path. One file per run, dated."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = run.dataset.replace("/", "-").removesuffix(".jsonl")
    base = f"{stem}-{run.started_at:%Y%m%d-%H%M}"
    path = directory / f"{base}.md"
    taken = 1
    while path.exists():  # two runs in one minute keep both reports
        taken += 1
        path = directory / f"{base}-{taken}.md"
    path.write_text(to_markdown(run), encoding="utf-8")
    return path
