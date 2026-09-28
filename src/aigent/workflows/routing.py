"""Routing: classify the input, then hand it to the handler built for its kind.

uv run python -m aigent.workflows.routing ["question"] [--yes]

A small model picks the route; each route has its own prompt and its own model, and the route's
report scopes the search. A question the reports cannot answer is declined in code, for nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from aigent.config import MAX_TOKENS_ROUTE, MAX_TOKENS_ROUTED, MODEL, SMALL_MODEL
from aigent.config import THINKING_WORKFLOW_PARAM as THINKING
from aigent.llm import Llm, Request
from aigent.retrieval.chunk import context_block
from aigent.workflows.demo import run_demo
from aigent.workflows.reports import CATALOGUE, DocId, Search

QUESTIONS = (
    "What dividend per share did Reliance's board recommend for FY25?",
    "How does Tata Motors describe the trade-offs in its EV strategy?",
    "What will the RBI do with interest rates next quarter?",
)

ROUTER = f"You route questions about three annual reports:\n{CATALOGUE}"
LOOKUP = (
    "You answer a factual question from annual-report passages in one or two sentences. "
    "Cite the passage id in square brackets. If the passages do not say, say so."
)
ANALYSIS = (
    "You answer an analytical question from annual-report passages in under 150 words. Weigh "
    "what the passages say, cite passage ids in square brackets, and name what they leave out."
)
DECLINED = "That is not something these three annual reports can answer."


class Route(BaseModel):
    kind: Literal["lookup", "analysis", "out_of_scope"] = Field(
        description="lookup: one fact or figure from one report. analysis: needs explanation, "
        "comparison or several facts combined. out_of_scope: these reports cannot answer it."
    )
    report: DocId | None = Field(description="The one report it is about, else null.")
    reason: str = Field(description="One short sentence.")


@dataclass(frozen=True)
class Handler:
    system: str
    model: str


HANDLERS = {"lookup": Handler(LOOKUP, SMALL_MODEL), "analysis": Handler(ANALYSIS, MODEL)}


@dataclass(frozen=True)
class Routed:
    route: Route
    answer: str
    model: str | None


def run(llm: Llm, search: Search, question: str) -> Routed:
    ask = Request.ask(
        "route", ROUTER, question, MAX_TOKENS_ROUTE, model=SMALL_MODEL, thinking=THINKING
    )
    route = llm.record(ask, Route)
    handler = HANDLERS.get(route.kind)
    if handler is None:
        return Routed(route, DECLINED, None)
    prompt = f"{context_block(search(question, doc_id=route.report))}\n\nquestion: {question}"
    answer = Request.ask(
        route.kind,
        handler.system,
        prompt,
        MAX_TOKENS_ROUTED,
        model=handler.model,
        thinking=THINKING,
    )
    return Routed(route, llm.text(answer), handler.model)


def show(result: Routed) -> str:
    route = result.route
    where = f" in {route.report}" if route.report else ""
    handled = f"answered by {result.model}" if result.model else "declined, no call"
    return f"route: {route.kind}{where} ({route.reason}) — {handled}\n\n{result.answer}"


def main(argv: list[str] | None = None) -> None:
    run_demo(argv, prog="aigent.workflows.routing", questions=QUESTIONS, run=run, show=show)


if __name__ == "__main__":
    main()
