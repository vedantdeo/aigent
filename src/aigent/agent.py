"""The first agent: the model decides which searches to run, and when it has enough.

uv run python -m aigent.agent ["question"] [--yes]

The workflows fix the path in code. Here the model gets the reports, the calculator and the web as
tools, and loops on the SDK's tool runner through `llm`, which admits every turn before it is sent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import cast

from aigent.config import MAX_AGENT_TURNS, MAX_USD_PER_RUN
from aigent.config import MAX_TOKENS_TOOL_LOOP as MAX_TOKENS
from aigent.errors import BudgetExceeded, TurnsExhausted
from aigent.llm import Llm, Request
from aigent.messages import Block, Reply
from aigent.report_tools import SEARCH_TOOL, ReportSearch, Searched
from aigent.tools import ALL_TOOLS, WEB_SEARCH_TOOL, execute_tool
from aigent.workflows.demo import run_demo
from aigent.workflows.reports import CATALOGUE, Search

QUESTIONS = (
    "Which of the three companies is most exposed to a slowdown in rural demand, and why?",
    "What dividend per share did Reliance's board recommend for FY25?",
    "How has Reliance's share price moved since it announced its FY25 results?",
)

SYSTEM = (
    f"You answer questions about three FY25 annual reports:\n{CATALOGUE}\n\n"
    "Find what you need with search_reports: search as often as the question needs, rephrase when "
    "the passages miss, and search one report when the question names a company. Cite passage ids "
    "in square brackets, and say what the reports do not cover. Use calculate for arithmetic. Use "
    "web_search only for what the reports cannot hold, such as events and prices after them."
)

FINISH = (
    "You are out of searches: do not call any more tools. Answer now from the passages you have, "
    "cite them, and say plainly what you could not check."
)

TOOLS = [SEARCH_TOOL, *ALL_TOOLS, WEB_SEARCH_TOOL]
CALLS = ("tool_use", "server_tool_use")


def called(reply: Reply) -> list[str]:
    """The tools a turn asked for, ours and the server's, in the order it asked."""
    return [str(block["name"]) for block in reply.blocks if block.get("type") in CALLS]


@dataclass(frozen=True)
class WebSources:
    """What a run's web searches left in its turns: pages cited, other pages returned, and how many
    searches ran."""

    cited: list[str] = field(default_factory=list[str])
    returned: list[str] = field(default_factory=list[str])
    searches: int = 0


@dataclass(frozen=True)
class Answered:
    """`stopped` says why searching ended early, if it did; `answer` is None when there was no
    room left even to answer."""

    answer: str | None
    searches: list[Searched]
    turns: int
    stopped: str | None = None
    web: WebSources = field(default_factory=WebSources)
    tools: list[str] = field(default_factory=list[str])  # every tool called, in order


def run(llm: Llm, search: Search, question: str, model: str | None = None) -> Answered:
    reports = ReportSearch(search)

    def dispatch(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        if name == SEARCH_TOOL["name"]:
            return (reports(arguments), False)
        return execute_tool(name, arguments)

    request = Request(
        "agent",
        [{"role": "user", "content": question}],
        MAX_TOKENS,
        system=SYSTEM,
        model=model,
        tools=TOOLS,
        thinking={"type": "adaptive"},
        cache_control={"type": "ephemeral"},
    )
    before = len(llm.trace)
    try:
        ran = llm.run_tools(request, dispatch, max_turns=MAX_AGENT_TURNS, finish=FINISH)
    except (BudgetExceeded, TurnsExhausted) as stop:
        # No room even to answer: what was searched so far is all there is to show for it.
        return Answered(None, reports.searches, len(llm.trace) - before, stopped=str(stop))
    answer = ran.message.text
    turns = len(llm.trace) - before
    tools = [name for turn in ran.turns for name in called(turn)]
    return Answered(answer, reports.searches, turns, ran.cut_short, _web_sources(ran.turns), tools)


def cited(blocks: Iterable[Block]) -> list[str]:
    """Every web page a turn's text cites through the API's structured citations, once each."""
    urls: dict[str, None] = {}
    for block in blocks:
        if block.get("type") == "text":
            for citation in cast(Sequence[Block], block.get("citations") or []):
                if citation.get("type") == "web_search_result_location":
                    urls[cast(str, citation["url"])] = None
    return list(urls)


def _web_sources(turns: Sequence[Reply]) -> WebSources:
    """Every page cited or returned in any turn, not just the last: a model answers after it
    searches, often turns later."""
    pages = cited(block for message in turns for block in message.blocks)
    returned: dict[str, None] = {}
    for message in turns:
        for block in message.blocks:
            if block.get("type") == "web_search_tool_result":
                results = block.get("content")
                if isinstance(results, list):
                    found = cast(Sequence[Block], results)
                    returned.update(dict.fromkeys(cast(str, page["url"]) for page in found))
    searches = sum(message.usage.web_searches for message in turns)
    return WebSources(pages, [url for url in returned if url not in pages], searches)


def show(result: Answered) -> str:
    lines = [f"{result.turns} turns, {len(result.searches)} searches:"]
    for searched in result.searches:
        found = ", ".join(searched.found) or "nothing"
        lines.append(f"  search_reports({searched.query!r}, {searched.report}) -> {found}")
    if result.stopped is not None:
        lines.append(f"\nsearching stopped: {result.stopped}")
    if result.answer is not None:
        lines.append(f"\n{result.answer}")
    web = result.web
    if web.cited:
        lines += ["\nweb pages cited:", *(f"- {url}" for url in web.cited)]
    if web.returned:
        lines += ["\nweb pages searched and not cited:", *(f"- {url}" for url in web.returned)]
    if web.searches and not (web.cited or web.returned):
        lines.append(f"\n{web.searches} web searches ran, and no page links came back with them.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    run_demo(
        argv,
        prog="aigent.agent",
        questions=QUESTIONS,
        run=run,
        show=show,
        limit_usd=MAX_USD_PER_RUN,
        scope="run",
    )


if __name__ == "__main__":
    main()
