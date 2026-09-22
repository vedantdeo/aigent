"""The first agent: the model decides which searches to run, and when it has enough.

uv run python -m entropic.agent ["question"] [--yes]

The workflows fix the path in code. Here the model gets the reports as a tool beside the calculator
and loops on the SDK's tool runner — through `llm`, so every turn is admitted before it is sent —
until it answers. Thinking stays on: with it off, a tool call can come out as plain text. The
conversation is cached, so each turn reads what the last one sent rather than paying for it again.
Out of turns or budget, it is told to stop searching and answer from what it has.
"""

from __future__ import annotations

from dataclasses import dataclass

from anthropic.types import ToolParam

from entropic.config import MAX_AGENT_TURNS, MAX_USD_PER_RUN
from entropic.config import MAX_TOKENS_TOOL_LOOP as MAX_TOKENS
from entropic.errors import BudgetExceeded, TurnsExhausted
from entropic.llm import Llm, Request
from entropic.retrieval.chunk import context_block
from entropic.tools import ALL_TOOLS, execute_tool
from entropic.workflows.demo import run_demo
from entropic.workflows.reports import CATALOGUE, REPORTS, Search

QUESTIONS = (
    "Which of the three companies is most exposed to a slowdown in rural demand, and why?",
    "What dividend per share did Reliance's board recommend for FY25?",
)

SYSTEM = (
    f"You answer questions about three FY25 annual reports:\n{CATALOGUE}\n\n"
    "Find what you need with search_reports: search as often as the question needs, rephrase when "
    "the passages miss, and search one report when the question names a company. Cite passage ids "
    "in square brackets, and say what the reports do not cover. Use calculate for arithmetic."
)

FINISH = (
    "You are out of searches: do not call any more tools. Answer now from the passages you have, "
    "cite them, and say plainly what you could not check."
)

SEARCH_TOOL: ToolParam = {
    "name": "search_reports",
    "description": (
        "Search the annual reports for passages that answer a query. Returns up to five passages, "
        "each tagged with the id to cite it by. Different words find different passages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in a report's words."},
            "report": {
                "type": "string",
                "enum": [*REPORTS, "all"],
                "description": "One report's id to search only it, or all.",
            },
        },
        "required": ["query", "report"],
        "additionalProperties": False,
    },
    "strict": True,
}
TOOLS = [SEARCH_TOOL, *ALL_TOOLS]


@dataclass(frozen=True)
class Searched:
    query: str
    report: str
    found: list[str]


@dataclass(frozen=True)
class Answered:
    """`stopped` says why searching ended early, if it did; `answer` is None when there was no
    room left even to answer."""

    answer: str | None
    searches: list[Searched]
    turns: int
    stopped: str | None = None


def run(llm: Llm, search: Search, question: str) -> Answered:
    searches: list[Searched] = []

    def dispatch(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        if name != SEARCH_TOOL["name"]:
            return execute_tool(name, arguments)
        query, report = str(arguments.get("query", "")), str(arguments.get("report", "all"))
        passages = search(query, doc_id=None if report == "all" else report)
        searches.append(Searched(query, report, [chunk.id for chunk in passages]))
        return (context_block(passages) or "No passages matched that query.", False)

    request = Request(
        "agent",
        [{"role": "user", "content": question}],
        MAX_TOKENS,
        system=SYSTEM,
        tools=TOOLS,
        thinking={"type": "adaptive"},
        cache_control={"type": "ephemeral"},
    )
    before = len(llm.trace)
    try:
        ran = llm.run_tools(request, dispatch, max_turns=MAX_AGENT_TURNS, finish=FINISH)
    except (BudgetExceeded, TurnsExhausted) as stop:
        # No room even to answer: what was searched so far is all there is to show for it.
        return Answered(None, searches, len(llm.trace) - before, stopped=str(stop))
    answer = "".join(block.text for block in ran.message.content if block.type == "text")
    return Answered(answer, searches, len(llm.trace) - before, stopped=ran.cut_short)


def show(result: Answered) -> str:
    lines = [f"{result.turns} turns, {len(result.searches)} searches:"]
    for searched in result.searches:
        found = ", ".join(searched.found) or "nothing"
        lines.append(f"  search_reports({searched.query!r}, {searched.report}) -> {found}")
    if result.stopped is not None:
        lines.append(f"\nsearching stopped: {result.stopped}")
    if result.answer is not None:
        lines.append(f"\n{result.answer}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    run_demo(
        argv,
        prog="entropic.agent",
        questions=QUESTIONS,
        run=run,
        show=show,
        limit_usd=MAX_USD_PER_RUN,
        scope="run",
    )


if __name__ == "__main__":
    main()
