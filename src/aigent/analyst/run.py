"""Questions through the capstone's graph, live, each written step by step to its own Markdown
transcript under `runs/analyst/`. Estimates and exits unless `--yes`.

    uv run python -m aigent.analyst.run --generalist sarvam-thinking --thinking off --yes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aigent import tracing
from aigent.analyst.cfg_analyst import (
    ANALYST_SERVED_TOOLS,
    ANALYST_WORKERS,
    CRITIC_MODEL,
    GENERALIST_MODEL,
    MAX_USD_ANALYST_CRITIC,
    MAX_USD_ANALYST_GENERALIST,
    MAX_USD_ANALYST_SPECIALIST,
)
from aigent.analyst.experts import BUILDERS, experts
from aigent.analyst.graph import Analysed, run
from aigent.analyst.router import Router
from aigent.analyst.served import warm_up
from aigent.analyst.toolbox import Toolbox
from aigent.llm.adapters import CLIENTS, spec
from aigent.llm.core import Llm
from aigent.llm.messages import Thinking
from aigent.llm.pricing import PRICES, Budget, cost_usd
from aigent.mcp_client import OWN_SERVER, connect
from aigent.transcript import FileTracer

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs" / "analyst"
# Twenty questions written so the specialists have real work: tables, several years, headlines.
PROBE = Path(__file__).resolve().parents[3] / "evals" / "datasets" / "analyst-probe.jsonl"

# The questions the graph has been run on since 2026-10-13, asked when none is given.
QUESTIONS = (
    "How much did ITC's dividend per share grow from FY24 to FY25, and what did its FY25 annual "
    "report say about the dividend?",
    "What capital expenditure did Reliance report for FY25, and how did it compare with FY24?",
    "What free cash flow did Jaguar Land Rover report in Tata Motors' FY25 annual report?",
)


@dataclass(frozen=True)
class Generalist:
    """Who researches and writes: a client, the model it is asked as (None for the client's own),
    and how it may think."""

    client: str
    model: str | None
    thinking: Thinking | None

    @property
    def label(self) -> str:
        return self.model or spec(self.client).model

    @property
    def calm(self) -> bool:
        """Whether it stops reasoning once a specialist has answered: on the OpenAI wire, where a
        reasoning server's reasoning can be switched off per turn."""
        return spec(self.client).wire == "openai"


def generalist_of(named: str, thinking: str) -> Generalist:
    """`client[:model]` and `adaptive`, `off` or a token budget, as a Generalist. ValueError for
    an unknown client or a budget that is not a number."""
    client, _, model = named.partition(":")
    if client not in CLIENTS:
        raise ValueError(f"no client named {client!r}; there is {', '.join(sorted(CLIENTS))}")
    chosen: Thinking | None
    if thinking == "adaptive":
        chosen = {"type": "adaptive"}
    elif thinking == "off":
        chosen = None
    elif thinking.isdigit():
        chosen = {"type": "enabled", "budget_tokens": int(thinking)}
    else:
        raise ValueError(f"thinking is adaptive, off or a token budget, not {thinking!r}")
    return Generalist(client, model or None, chosen)


def served_by(client: str, server: str) -> str:
    """The client that reaches `client`'s specialist on `server` (`hostllm` or `vllm`).
    ValueError if there is none."""
    named = client.replace("hostllm-", f"{server}-", 1)
    if named not in CLIENTS:
        raise ValueError(f"no client serves {client} on {server}")
    return named


def load_questions(path: Path) -> list[tuple[str, str]]:
    """Each row's id and question, in file order; `//` lines are comments."""
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    found = [json.loads(line) for line in rows if not line.startswith("//")]
    return [(str(row["id"]), str(row["question"])) for row in found]


def in_parallel[Item, Out](
    items: Sequence[Item], work: Callable[[Item], Out], workers: int
) -> list[Out | Exception]:
    """`work` on every item, `workers` at a time on threads of their own; results in item order,
    with what an item raised in place of its result."""

    def guarded(item: Item) -> Out | Exception:
        try:
            return work(item)
        except Exception as failed:
            return failed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(guarded, items))


def worst_case_usd(questions: int) -> float:
    """The most a run can spend: every question's ceilings, its specialists' included."""
    each = MAX_USD_ANALYST_GENERALIST + MAX_USD_ANALYST_CRITIC
    return questions * (each + len(BUILDERS) * MAX_USD_ANALYST_SPECIALIST)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _outcome(done: Analysed) -> str:
    if done.memo is None:
        return f"no memo: {done.stopped}"
    verdicts = [done.verdicts[c.id].verdict for c in done.memo.claims if c.id in done.verdicts]
    supported = sum(v == "supported" for v in verdicts)
    return f"{supported}/{len(done.memo.claims)} claims supported, {done.revisions} redrafts"


def result(done: Analysed, generalist: Llm, critic: Llm) -> list[str]:
    """What a run came to, as Markdown paragraphs: the memo with each claim's verdict, every
    routing, and the spend."""
    lines = ["## Result", f"**Outcome:** {_outcome(done)}"]
    if done.stopped:
        lines.append(f"**Stopped:** {done.stopped}")
    if done.memo is not None:
        lines.append(f"**Summary:** {done.memo.summary}")
        for claim in done.memo.claims:
            judged = done.verdicts.get(claim.id)
            verdict = f"**{judged.verdict}** ({judged.reason})" if judged else "not judged"
            lines.append(f"- {claim.id} {claim.text} `{', '.join(claim.citations)}` — {verdict}")
        if done.memo.caveats:
            lines.append("**Caveats:** " + " ".join(done.memo.caveats))
    for r in done.routed:
        lines.append(
            f"- routed {r.specialist}: answered by the {r.path}, {r.seconds:.1f} s, specialist "
            f"${r.specialist_usd:.4f}, generalist for the same tokens ${r.counterfactual_usd:.4f}"
            + (f"; rejected because {r.failure}" if r.failure else "")
        )
    lines.append(
        f"**Spent:** generalist ${generalist.spent_usd:.4f}, critic ${critic.spent_usd:.4f}"
    )
    return lines


def analyse(
    question: str,
    chosen: Generalist,
    generalist: Llm,
    critic: Llm,
    toolbox: Toolbox,
    path: Path,
) -> Analysed:
    """One question through the graph, every step of it written to `path` as it happens."""
    tracer = FileTracer(path)
    tracer.write(
        f"# {question}",
        f"Generalist {chosen.label} on the {chosen.client} client, thinking {chosen.thinking}; "
        f"critic {CRITIC_MODEL}; {datetime.now():%Y-%m-%d %H:%M}",
    )
    try:
        with tracing.using(tracer):
            done = run(
                generalist, critic, toolbox, question, chosen.model, chosen.thinking, chosen.calm
            )
        tracer.write(*result(done, generalist, critic))
        return done
    finally:
        tracer.close()


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aigent.analyst.run", description=__doc__)
    parser.add_argument("--yes", action="store_true", help="send the calls; without, estimate")
    parser.add_argument(
        "--question", action="append", help="ask this; repeat for more (default: the three)"
    )
    parser.add_argument(
        "--questions", type=Path, help=f"a JSONL file of id and question, such as {PROBE.name}"
    )
    parser.add_argument(
        "--generalist",
        default=f"anthropic:{GENERALIST_MODEL}",
        help="client[:model] that researches and writes",
    )
    parser.add_argument(
        "--thinking", default="adaptive", help="adaptive, off, or a thinking budget in tokens"
    )
    parser.add_argument("--out", type=Path, default=RUNS_DIR, help="where transcripts go")
    parser.add_argument(
        "--workers", type=int, default=ANALYST_WORKERS, help="questions in flight at once"
    )
    parser.add_argument(
        "--specialists",
        default="hostllm",
        choices=("hostllm", "vllm"),
        help="the engine that serves the fine-tunes: our own, or vLLM",
    )
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # a tokenizer's load notes
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        chosen = generalist_of(args.generalist, args.thinking)
    except ValueError as bad:
        raise SystemExit(str(bad)) from None
    if unpriced := [m for m in (chosen.label, CRITIC_MODEL) if m not in PRICES]:
        raise SystemExit(f"{', '.join(unpriced)} has no price; add it to pricing.PRICES")
    if args.question:
        questions = [(f"q{n}", q) for n, q in enumerate(args.question, 1)]
    elif args.questions:
        questions = load_questions(args.questions)
    else:
        questions = [(f"q{n}", q) for n, q in enumerate(QUESTIONS, 1)]
    print(
        f"{len(questions)} questions: generalist {chosen.label} ({chosen.client}), thinking "
        f"{chosen.thinking}; critic {CRITIC_MODEL}; specialists on {args.specialists}"
    )
    print(f"worst case ${worst_case_usd(len(questions)):.2f}; {args.workers} at a time")
    if not args.yes:
        print("estimate only: nothing sent. Add --yes to run.")
        return

    began = time.monotonic()
    warm_up(Llm(served_by("hostllm-analyst", args.specialists)))
    print(f"specialists warm after {time.monotonic() - began:.0f} s")
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(chosen.label)}"
    with connect(OWN_SERVER, only=ANALYST_SERVED_TOOLS, fenced=False) as served:

        def ask(item: tuple[str, str]) -> Analysed:
            named, question = item
            generalist = Llm(chosen.client, budget=Budget(limit_usd=MAX_USD_ANALYST_GENERALIST))
            small = {
                c: Llm(
                    served_by(c, args.specialists),
                    budget=Budget(limit_usd=MAX_USD_ANALYST_SPECIALIST),
                )
                for c in BUILDERS
            }
            critic = Llm("anthropic", budget=Budget(limit_usd=MAX_USD_ANALYST_CRITIC))
            toolbox = Toolbox(
                served.tools,
                served.dispatch,
                Router(price=lambda i, o: cost_usd(chosen.label, i, o)),
                experts(generalist, small=small.__getitem__, model=chosen.model),
            )
            path = args.out / f"{stamp}-{named}.md"
            try:
                done = analyse(question, chosen, generalist, critic, toolbox, path)
            finally:
                spent = generalist.spent_usd + critic.spent_usd
                fees = sum(s.spent_usd for s in small.values())
                print(f"{named} done, ${spent:.4f}, specialists ${fees:.4f}: {path}", flush=True)
            print(f"{named}: {_outcome(done)}", flush=True)
            return done

        answered = in_parallel(questions, ask, args.workers)
    for (named, _), done in zip(questions, answered, strict=True):
        if isinstance(done, Exception):
            print(f"{named} failed: {type(done).__name__}: {done}")
    print(f"{time.monotonic() - began:.0f} s in all")


if __name__ == "__main__":
    main()
