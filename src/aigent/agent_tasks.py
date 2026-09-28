"""Project 2's eval: 25 multi-step tasks through the LangGraph agent, graded on how it got there.

uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5    # worst case only
uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5 --sample 5 --yes
uv run --group graph python -m aigent.agent_tasks --regrade evals/reports/<run>.rows.jsonl --yes

Four graders: the right tools, in a sane order, a finished answer, and a correct one by the judge.
A task passes when all four do; the report gives that rate and what each task cost. Every run
saves its answers beside the report, so `--regrade` can grade them again without the agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from aigent.adapters import CLIENTS, spec
from aigent.agent_graph import Traced
from aigent.config import CLIENT, MAX_USD_PER_EVAL, MAX_USD_PER_TASK, max_tokens
from aigent.evals.dataset import Case, digest, load_jsonl
from aigent.evals.grade import Grader, Outcome, flag, right_tools, tool_order
from aigent.evals.judge import LlmJudge
from aigent.evals.report import write_report
from aigent.evals.runner import Task, run_eval
from aigent.llm import Llm
from aigent.messages import Usage
from aigent.pricing import Budget, estimate_eval_usd, usage_cost
from aigent.retrieval.answer import spread
from aigent.workflows.demo import build_search
from aigent.workflows.reports import Search

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "evals" / "datasets" / "tasks.jsonl"

RUBRIC = """The reference says what a right answer holds: the figures from the reports, any result \
computed from them, or that the right answer is to decline.

Pass if the answer states the same figures and the same computed result, within ordinary rounding. \
Where the reference allows a range, or asks for a figure from a web source, pass an answer inside \
that range that attributes its figure to a source. A figure from the web is judged by whether it \
is attributed and used correctly, never against what you remember: the web is newer than you are. \
Where the reference says to decline, pass only an answer that declines or corrects the premise and \
invents nothing.

Fail an answer that gives a different figure, invents one, or declines where the reference has an \
answer."""


def total(usages: Sequence[Usage]) -> Usage:
    """One task's calls summed, so the runner bills the task as one row."""
    return Usage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cache_write_tokens=sum(u.cache_write_tokens for u in usages),
        cache_read_tokens=sum(u.cache_read_tokens for u in usages),
        web_searches=sum(u.web_searches for u in usages),
    )


AgentRun = Callable[..., Traced]


def agent_run(name: str) -> AgentRun:
    """The agent to run, imported only when chosen: each framework is an optional group."""
    if name == "adk":
        from aigent.agent_adk import run as adk

        return adk
    from aigent.agent_graph import run as graph

    return graph


def agent_task(
    search: Search,
    client: str = CLIENT,
    model: str | None = None,
    sdk: object | None = None,
    limit_usd: float = MAX_USD_PER_TASK,
    agent: str = "langgraph",
) -> Task:
    """Each case in a fresh `Llm` with its own ceiling, sharing one connection."""
    answering = model or spec(client).model

    def task(case: Case) -> Outcome:
        question = case.input.get("task")
        if not isinstance(question, str):
            return Outcome(error=f"case {case.id} has no task")
        llm = Llm(client, sdk=sdk, budget=Budget(limit_usd=limit_usd, scope="run"))
        try:
            traced = agent_run(agent)(llm, search, question, model=model)
        except Exception as failed:
            # The calls it made before failing were billed, so the row still carries them.
            spent = total([call.usage for call in llm.trace])
            return Outcome(error=f"{type(failed).__name__}: {failed}", usage=spent, model=answering)
        return Outcome(
            output={
                "answer": traced.answer,
                "tools": list(traced.tools),
                "finished": traced.answer is not None,
                "cut_short": traced.stopped,
                "turns": traced.turns,
            },
            raw=traced.answer if traced.answer is not None else "(no answer)",
            usage=total([call.usage for call in llm.trace]),
            model=answering,
        )

    return task


def recording(task: Task, into: dict[str, Outcome]) -> Task:
    """`task`, keeping every outcome it returns by case id."""

    def recorded(case: Case) -> Outcome:
        into[case.id] = outcome = task(case)
        return outcome

    return recorded


def save_rows(path: Path, outcomes: Mapping[str, Outcome]) -> None:
    """Each task's answer, trajectory and usage, sorted by id: what a re-grade reads back."""
    lines = [
        json.dumps(
            {
                "id": case_id,
                "output": outcome.output,
                "raw": outcome.raw,
                "error": outcome.error,
                "model": outcome.model,
                "usage": asdict(outcome.usage) if outcome.usage is not None else None,
            },
            ensure_ascii=False,
        )
        for case_id, outcome in sorted(outcomes.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_rows(path: Path) -> dict[str, tuple[Outcome, Usage | None]]:
    """A saved run: each outcome without its usage, since re-grading must not bill the agent
    again, and that usage beside it for the record."""
    saved: dict[str, tuple[Outcome, Usage | None]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        usage = Usage(**row["usage"]) if row["usage"] else None
        outcome = Outcome(
            output=row["output"], raw=row["raw"], error=row["error"], model=row["model"]
        )
        saved[row["id"]] = (outcome, usage)
    return saved


def replayed(saved: Mapping[str, tuple[Outcome, Usage | None]]) -> Task:
    """A task answering from a saved run."""

    def task(case: Case) -> Outcome:
        found = saved.get(case.id)
        return found[0] if found is not None else Outcome(error=f"no saved answer for {case.id}")

    return task


def graders(
    client: str = CLIENT, sdk: object | None = None, judge_model: str | None = None
) -> dict[str, Grader]:
    """Trajectory first, then the answer: the three free graders before the paid one."""
    return {
        "right_tools": right_tools(),
        "tool_order": tool_order(),
        "finished": flag("finished"),
        "correct": LlmJudge(
            rubric=RUBRIC, reference="reference", model=judge_model, client=client, sdk=sdk
        ),
    }


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aigent.agent_tasks", description=__doc__)
    parser.add_argument(
        "--yes", action="store_true", help="send the calls; without it, estimate only"
    )
    parser.add_argument("--sample", type=int, metavar="N", help="N tasks spread across the set")
    parser.add_argument("--only", help="comma-separated task ids, such as pt-015,pt-016")
    parser.add_argument("--model", help="the agent's model (default: the client's own)")
    parser.add_argument("--judge-model", help="the judge's model; never the agent's own")
    parser.add_argument("--client", default=CLIENT, choices=sorted(CLIENTS), help="who answers")
    parser.add_argument(
        "--limit", type=float, default=MAX_USD_PER_EVAL, help="the whole run's ceiling in USD"
    )
    parser.add_argument("--cache", action="store_true", help="reuse PDF text and chunk vectors")
    parser.add_argument(
        "--agent", default="langgraph", choices=("langgraph", "adk"), help="which build answers"
    )
    parser.add_argument("--regrade", type=Path, help="grade a saved run's answers, not a new run")
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    cases = load_jsonl(DATASET)
    if args.sample is not None:
        cases = spread(cases, args.sample)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [case for case in cases if case.id in wanted]
        if unknown := wanted - {case.id for case in cases}:
            raise SystemExit(f"no task with id {', '.join(sorted(unknown))}")
    settings = spec(args.client)
    model = args.model or settings.model
    judge = args.judge_model or settings.judge_model
    if judge == model:
        raise SystemExit(f"{model} would grade its own answers; pass a different --judge-model")

    # Each task is held to its own ceiling, so that bounds its cost; the judge is one call a task.
    judge_cap = max_tokens("JUDGE", settings)
    judged = estimate_eval_usd(judge, len(cases), 2_000, judge_cap)
    if args.regrade is not None:
        regrade(args, cases, judge, judged)
        return
    worst = len(cases) * MAX_USD_PER_TASK + judged
    print(f"{DATASET.name}: {len(cases)} tasks, {args.agent} agent on {model}, judged by {judge}")
    print(f"worst case ${worst:.2f}: ${MAX_USD_PER_TASK:.2f} a task at most, ${judged:.2f} judging")
    print(f"run ceiling ${args.limit:.2f}")
    if not args.yes:
        print(
            "\nnothing spent. re-run with --yes (and --limit at or above the worst case) to send."
        )
        return

    shared = Llm.for_eval(args.client)
    search = build_search(args.cache)
    outcomes: dict[str, Outcome] = {}
    task = agent_task(search, args.client, args.model, shared.sdk, agent=args.agent)
    result = run_eval(
        cases,
        {args.agent: recording(task, outcomes)},
        graders(args.client, shared.sdk, judge),
        dataset=DATASET.name,
        digest=digest(DATASET),
        model=model,
        limit_usd=args.limit,
        worst_usd=worst,
    )
    rows = result.rows
    mean = result.spent_usd / len(rows) if rows else 0.0
    passed = sum(row.passed for row in rows)
    print(f"\nsuccess {passed}/{len(rows)}, mean ${mean:.4f} a task")
    report = write_report(result)
    save_rows(report.with_suffix(".rows.jsonl"), outcomes)
    print(f"report: {report}\nanswers: {report.with_suffix('.rows.jsonl')}")


def regrade(args: argparse.Namespace, cases: Sequence[Case], judge: str, judged: float) -> None:
    """Grade a saved run again: the free graders and the judge, and nothing sent to the agent."""
    saved = load_rows(args.regrade)
    cases = [case for case in cases if case.id in saved]
    print(f"re-grading {len(cases)} saved answers from {args.regrade.name}, judged by {judge}")
    print(f"worst case ${judged:.2f}, all of it judging")
    if not args.yes:
        print("\nnothing spent. re-run with --yes to send the judge's calls.")
        return
    shared = Llm.for_eval(args.client)
    model = next((o.model for o, _ in saved.values() if o.model), "unknown")
    result = run_eval(
        cases,
        {args.agent: replayed(saved)},
        graders(args.client, shared.sdk, judge),
        dataset=DATASET.name,
        digest=digest(DATASET),
        model=model,
        limit_usd=args.limit,
        worst_usd=judged,
    )
    agent = sum(usage_cost(model, u) for _, u in saved.values() if u is not None) / len(saved)
    passed = sum(row.passed for row in result.rows)
    print(f"\nsuccess {passed}/{len(result.rows)}; the agent had cost ${agent:.4f} a task")
    print(f"report: {write_report(result)}")


if __name__ == "__main__":
    main()
