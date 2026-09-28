"""Project 2's eval: 25 multi-step tasks through the LangGraph agent, graded on how it got there.

uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5    # worst case only
uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5 --sample 5 --yes

Four graders: the right tools, in a sane order, a finished answer, and a correct one by the judge.
A task passes when all four do; the report gives that rate and what each task cost.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from aigent.adapters import CLIENTS, spec
from aigent.agent_graph import run
from aigent.config import CLIENT, MAX_USD_PER_EVAL, MAX_USD_PER_TASK, max_tokens
from aigent.evals.dataset import Case, digest, load_jsonl
from aigent.evals.grade import Grader, Outcome, flag, right_tools, tool_order
from aigent.evals.judge import LlmJudge
from aigent.evals.report import write_report
from aigent.evals.runner import Task, run_eval
from aigent.llm import Llm
from aigent.messages import Usage
from aigent.pricing import Budget, estimate_eval_usd
from aigent.retrieval.answer import spread
from aigent.workflows.demo import build_search
from aigent.workflows.reports import Search

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "evals" / "datasets" / "tasks.jsonl"

RUBRIC = """The reference says what a right answer holds: the figures from the reports, any result \
computed from them, or that the right answer is to decline.

Pass if the answer states the same figures and the same computed result, within ordinary rounding. \
Where the reference allows a range, or asks for a figure from a web source, pass an answer inside \
that range that attributes its figure to a source. Where the reference says to decline, pass only \
an answer that declines or corrects the premise and invents nothing.

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


def agent_task(
    search: Search,
    client: str = CLIENT,
    model: str | None = None,
    sdk: object | None = None,
    limit_usd: float = MAX_USD_PER_TASK,
) -> Task:
    """Each case in a fresh `Llm` with its own ceiling, sharing one connection."""
    answering = model or spec(client).model

    def task(case: Case) -> Outcome:
        question = case.input.get("task")
        if not isinstance(question, str):
            return Outcome(error=f"case {case.id} has no task")
        llm = Llm(client, sdk=sdk, budget=Budget(limit_usd=limit_usd, scope="run"))
        try:
            traced = run(llm, search, question, model=model)
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
    parser.add_argument("--model", help="the agent's model (default: the client's own)")
    parser.add_argument("--judge-model", help="the judge's model; never the agent's own")
    parser.add_argument("--client", default=CLIENT, choices=sorted(CLIENTS), help="who answers")
    parser.add_argument(
        "--limit", type=float, default=MAX_USD_PER_EVAL, help="the whole run's ceiling in USD"
    )
    parser.add_argument("--cache", action="store_true", help="reuse PDF text and chunk vectors")
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    cases = load_jsonl(DATASET)
    if args.sample is not None:
        cases = spread(cases, args.sample)
    settings = spec(args.client)
    model = args.model or settings.model
    judge = args.judge_model or settings.judge_model
    if judge == model:
        raise SystemExit(f"{model} would grade its own answers; pass a different --judge-model")

    # Each task is held to its own ceiling, so that bounds its cost; the judge is one call a task.
    judge_cap = max_tokens("JUDGE", settings)
    judged = estimate_eval_usd(judge, len(cases), 2_000, judge_cap)
    worst = len(cases) * MAX_USD_PER_TASK + judged
    print(f"{DATASET.name}: {len(cases)} tasks, agent on {model}, judged by {judge}")
    print(f"worst case ${worst:.2f}: ${MAX_USD_PER_TASK:.2f} a task at most, ${judged:.2f} judging")
    print(f"run ceiling ${args.limit:.2f}")
    if not args.yes:
        print(
            "\nnothing spent. re-run with --yes (and --limit at or above the worst case) to send."
        )
        return

    shared = Llm.for_eval(args.client)
    search = build_search(args.cache)
    result = run_eval(
        cases,
        {"langgraph": agent_task(search, args.client, args.model, shared.sdk)},
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
    print(f"report: {write_report(result)}")


if __name__ == "__main__":
    main()
