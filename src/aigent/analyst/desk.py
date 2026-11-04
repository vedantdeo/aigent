"""The analyst desk: the capstone's graph trimmed to the facts database, on a local web page. A
question is researched with `fundamentals` and the metric tools only, written as a memo of cited
claims and checked claim by claim by the critic; each computed figure a supported claim cites is
written back as `derived-ai` under its metric's name. The page shows each claim's verdict, the
facts it cites with their tier, what was written back, the tool calls, and the spend.

    uv run --group desk python -m aigent.analyst.desk --yes
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anthropic.types import ToolParam

from aigent import guardrails
from aigent.analyst.cfg_analyst import (
    CRITIC_MODEL,
    DESK_HOST,
    DESK_PORT,
    DESK_SERVED_TOOLS,
    GENERALIST_MODEL,
    INDIAN_RUPEE_UNITS,
    MAX_USD_ANALYST_CRITIC,
    MAX_USD_ANALYST_GENERALIST,
)
from aigent.analyst.graph import Analysed, run
from aigent.analyst.metric_tools import METRIC_TOOLS, MetricTools, chained, write_back
from aigent.analyst.router import Router
from aigent.analyst.run import Generalist, generalist_of
from aigent.analyst.specialists import Passage
from aigent.analyst.toolbox import LOOKUP, Toolbox
from aigent.errors import BudgetExceeded, GuardrailTripped
from aigent.facts.derived_concepts import seed
from aigent.facts.store import FACTS_DB, Writer, connect
from aigent.llm.core import Dispatch, Llm
from aigent.llm.pricing import PRICES, Budget, cost_usd

if TYPE_CHECKING:
    from fastapi import FastAPI

PAGE = Path(__file__).with_name("desk.html")

_SCALE = ", ".join(f"{name} (10^{len(str(size)) - 1})" for name, size in INDIAN_RUPEE_UNITS)
RUPEES = (
    "State every rupee amount in Indian units, the largest of these it reaches: "
    f"{_SCALE}; below a thousand, in plain rupees. Write it as Rs with Indian digit grouping, "
    "such as Rs 58,130.76 crore or Rs 4.5 lakh, and never in millions or billions."
)

DESK_PROMPT = (
    "You are a financial analyst answering a question about Indian listed companies from a "
    "database of the figures in their results filings. Look each figure up with fundamentals, "
    "naming the company, the financial year (2025 is FY 2024-25), the period and the basis; it "
    "returns only verified figures, each with a cite_as, and says how many unverified entries it "
    "held back. A figure that needs arithmetic, such as a growth rate, a ratio or a margin, is "
    "a metric: plan it with plan_metric, fetch each fact its plan lists with fundamentals, "
    "exactly as listed, then compute it with compute_metric, which returns it with a cite_as of "
    "its own. plan_metric takes a time, a financial year and quarter meaning that quarter's "
    "end, and a period, the span ending then, or none for a snapshot such as a balance-sheet "
    "ratio; a metric comparing two times, such as a growth rate, takes the previous time and "
    "period too. Name them every time, as nothing is assumed, and state them in the answer. "
    "plan_metric plans only a metric that names every word you ask, or its exact "
    "name; otherwise it lists the closest in the company's scope: plan one by its exact name if "
    "it is the figure you need, else define one with define_metric in that scope, on a library "
    "formula over concepts or over metrics already defined. If no formula fits even so, propose "
    "one with propose_formula and say the figure could not be computed. "
    "Never do arithmetic yourself, never state a figure no tool returned, and never estimate "
    "one. When you have enough, answer in prose: state each fact with its figure, unit and "
    "period, and after it the cite_as of every fact or computed figure it rests on in square "
    "brackets. Only when a figure the question needs has no "
    "verified value, say so, and how many unverified entries fundamentals held back for it. "
    f"{RUPEES} {guardrails.UNTRUSTED_NOTE}"
)

Call = dict[str, object]
# A free local lookup at start, so the first question does not wait for the embedder to load.
WARM_UP: dict[str, object] = {
    "company": "ICICIBANK",
    "year": "2025",
    "what": "net profit",
    "period": "year",
    "basis": "standalone",
    "segment": None,
}


def recorded(dispatch: Dispatch, calls: list[Call]) -> Dispatch:
    """`dispatch`, each call appended to `calls` with its arguments, result and seconds."""

    def call(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        began = time.monotonic()
        content, is_error = dispatch(name, arguments)
        try:
            parsed: object = json.loads(content)
        except json.JSONDecodeError:
            parsed = content
        calls.append(
            {
                "tool": name,
                "arguments": arguments,
                "result": parsed,
                "error": is_error,
                "seconds": round(time.monotonic() - began, 2),
            }
        )
        return content, is_error

    return call


def cited(toolbox: Toolbox, ids: Sequence[str]) -> list[dict[str, object]]:
    """Each cited id's fact as `fundamentals` returned it, or a bare id the toolbox never saw."""
    out: list[dict[str, object]] = []
    for cite in ids:
        found = toolbox.evidence.get(cite)
        try:
            fact = json.loads(found.text) if found is not None else None
        except json.JSONDecodeError:
            fact = None
        out.append(cast(dict[str, object], fact) if isinstance(fact, dict) else {"cite_as": cite})
    return out


def held_back(calls: Sequence[Call]) -> int:
    """How many unverified entries `fundamentals` held back across the calls."""
    total = 0
    for call in calls:
        result = call["result"]
        if call["tool"] == LOOKUP and isinstance(result, dict):
            held = cast(dict[str, object], result).get("unverified_held")
            total += held if isinstance(held, int) else 0
    return total


def payload(
    question: str,
    done: Analysed | None,
    toolbox: Toolbox,
    calls: Sequence[Call],
    spent: dict[str, float],
    error: str | None = None,
    written: Sequence[dict[str, object]] = (),
    skipped: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """What the page shows for one question, as JSON."""
    answer: dict[str, object] = {
        "written": list(written),
        "skipped": list(skipped),
        "question": question,
        "calls": list(calls),
        "held_back": held_back(calls),
        "spent": {k: round(v, 4) for k, v in spent.items()},
        "error": error,
        "summary": None,
        "claims": [],
        "caveats": [],
        "draft": None,
        "stopped": None,
    }
    if done is None:
        return answer
    answer["draft"] = done.draft
    answer["stopped"] = done.stopped
    if done.memo is not None:
        answer["summary"] = done.memo.summary
        answer["caveats"] = list(done.memo.caveats)
        answer["claims"] = [
            {
                "id": claim.id,
                "text": claim.text,
                "verdict": judged.verdict if (judged := done.verdicts.get(claim.id)) else None,
                "reason": judged.reason if judged else None,
                "facts": cited(toolbox, claim.citations),
            }
            for claim in done.memo.claims
        ]
    return answer


def held_to(conn: sqlite3.Connection, today: date) -> str:
    """Today's date and the quarter end each company's latest results are held to, and how a
    question naming no time is read."""
    ends: dict[str, list[str]] = {}
    for ticker, end in conn.execute(
        "SELECT ticker, max(period_end) FROM filings GROUP BY ticker ORDER BY ticker"
    ):
        ends.setdefault(str(end), []).append(str(ticker))
    held = "; ".join(
        f"{_quarter_end(date.fromisoformat(end))} for {', '.join(tickers)}"
        for end, tickers in sorted(ends.items(), reverse=True)
    )
    return (
        f"Today is {today.day} {today:%B %Y}. The latest results the database holds end {held}. "
        "A question naming no time, such as one about now, asks about the latest time held: for "
        "a snapshot metric, as at that quarter's end (most companies other than banks file a "
        "balance sheet only at Q2 and Q4, so take the latest of those that holds it); for a flow, "
        "the latest quarter, 3M, unless the question means a year. Say in the answer which time "
        "you used and that later results are not held."
    )


def _quarter_end(end: date) -> str:
    """A quarter's last day as the financial quarter it closes: "Q4 FY2026 (31 March 2026)"."""
    quarter = {6: "Q1", 9: "Q2", 12: "Q3", 3: "Q4"}[end.month]
    year = end.year if end.month <= 3 else end.year + 1
    return f"{quarter} FY{year} ({end.day} {end:%B %Y})"


def answer(
    question: str,
    served_tools: Sequence[ToolParam],
    dispatch: Dispatch,
    chosen: Generalist,
    generalist: Llm,
    critic: Llm,
    facts: sqlite3.Connection,
    writer: Writer,
) -> dict[str, object]:
    """One question through the trimmed graph, as the page's JSON, its metric tools over `facts`
    and its writes as `writer`. A tripped guard or a spent budget comes back as `error`, with
    whatever was called before it; the computed figures supported claims cite are written back."""
    calls: list[Call] = []
    evidence: dict[str, Passage] = {}
    metrics = MetricTools(facts, evidence, writer, question)
    toolbox = Toolbox(
        [*served_tools, *METRIC_TOOLS],
        recorded(chained(metrics, dispatch), calls),
        Router(price=lambda i, o: cost_usd(chosen.label, i, o)),
        experts=[],
        evidence=evidence,
    )
    done: Analysed | None = None
    error: str | None = None
    try:
        done = run(
            generalist,
            critic,
            toolbox,
            question,
            chosen.model,
            chosen.thinking,
            chosen.calm,
            research_prompt=f"{DESK_PROMPT} {held_to(facts, date.today())}",
        )
    except (BudgetExceeded, GuardrailTripped) as stopped:
        error = f"{type(stopped).__name__}: {stopped}"
    spent = {"generalist": generalist.spent_usd, "critic": critic.spent_usd}
    written: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    if done is not None:
        written, skipped = write_back(facts, done, metrics, writer)
    return payload(question, done, toolbox, calls, spent, error, written, skipped)


def app(ask: Callable[[str], dict[str, object]]) -> FastAPI:
    """The page and `POST /ask`, one question at a time. FastAPI is imported here, so the module
    loads without the `desk` group."""
    from fastapi import Body, FastAPI
    from fastapi.responses import HTMLResponse

    served = FastAPI(title="Analyst desk")
    one_at_a_time = threading.Lock()

    @served.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    @served.post("/ask")
    def ask_one(question: str = Body(..., embed=True)) -> dict[str, object]:
        with one_at_a_time:
            return ask(question)

    return served


def worst_case_usd() -> float:
    """The most one question can spend: the generalist's ceiling and the critic's."""
    return MAX_USD_ANALYST_GENERALIST + MAX_USD_ANALYST_CRITIC


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aigent.analyst.desk", description=__doc__)
    parser.add_argument("--yes", action="store_true", help="serve live; without, estimate")
    parser.add_argument(
        "--generalist",
        default=f"anthropic:{GENERALIST_MODEL}",
        help="client[:model] that researches and writes",
    )
    parser.add_argument("--thinking", default="adaptive", help="adaptive, off, or a budget")
    parser.add_argument("--port", type=int, default=DESK_PORT)
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        chosen = generalist_of(args.generalist, args.thinking)
    except ValueError as bad:
        raise SystemExit(str(bad)) from None
    if unpriced := [m for m in (chosen.label, CRITIC_MODEL) if m not in PRICES]:
        raise SystemExit(f"{', '.join(unpriced)} has no price; add it to pricing.PRICES")
    print(
        f"generalist {chosen.label}, critic {CRITIC_MODEL}, tools {', '.join(DESK_SERVED_TOOLS)}; "
        f"each question at most ${worst_case_usd():.2f}"
    )
    if not args.yes:
        print("estimate only: nothing served. Add --yes to serve the page; every question is paid.")
        return

    import uvicorn

    from aigent.mcp_client import OWN_SERVER
    from aigent.mcp_client import connect as mcp_connect

    with mcp_connect(OWN_SERVER, only=DESK_SERVED_TOOLS, fenced=False) as served:
        began = time.monotonic()
        served.dispatch(LOOKUP, WARM_UP)
        print(f"lookup warm after {time.monotonic() - began:.0f} s", flush=True)
        started = datetime.now().astimezone()
        held = connect(FACTS_DB)
        try:
            refused = seed(
                held,
                Writer(
                    "seed|evals/reference/derived-concepts.jsonl",
                    started.isoformat(timespec="seconds"),
                    f"seed-{started:%Y%m%d-%H%M%S}",
                ),
            )
        finally:
            held.close()
        print(f"derived concepts seeded; refused: {refused or 'none'}", flush=True)

        def ask(question: str) -> dict[str, object]:
            generalist = Llm(chosen.client, budget=Budget(limit_usd=MAX_USD_ANALYST_GENERALIST))
            critic = Llm("anthropic", budget=Budget(limit_usd=MAX_USD_ANALYST_CRITIC))
            now = datetime.now().astimezone()
            writer = Writer(
                f"analyst-desk|{chosen.label}",
                now.isoformat(timespec="seconds"),
                f"desk-{now:%Y%m%d-%H%M%S}",
            )
            facts = connect(FACTS_DB)
            try:
                done = answer(
                    question,
                    served.tools,
                    served.dispatch,
                    chosen,
                    generalist,
                    critic,
                    facts,
                    writer,
                )
            finally:
                facts.close()
            spent = cast(dict[str, float], done["spent"])
            print(f"${sum(spent.values()):.4f}: {question}", flush=True)
            return done

        print(f"http://{DESK_HOST}:{args.port}", flush=True)
        uvicorn.run(app(ask), host=DESK_HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
