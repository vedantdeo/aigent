"""The report analyst wired to models: each job on the prompt its fine-tune is trained on, with a
search of one report and calculate, put to the served adapter and, as its fallback, to the
generalist; and the request that warms the endpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

from aigent.analyst.cfg_analyst import (
    GENERALIST_MODEL,
    SPECIALIST_TOOL_TURNS,
    SPECIALIST_TRIES,
    max_tokens,
)
from aigent.analyst.filings import COMPANIES
from aigent.analyst.specialists import (
    FigureJob,
    Found,
    Passage,
    Specialist,
    Told,
    check_figures,
    passages_in,
)
from aigent.analyst.toolbox import SEARCH
from aigent.errors import SpecialistUnreachable
from aigent.extraction.research import REPORT_SEARCH_TOOL, research_messages
from aigent.llm.adapters import spec
from aigent.llm.adapters.openai import UNREACHABLE
from aigent.llm.core import Dispatch, Llm, Request
from aigent.llm.messages import Block, Cache, Msg, Reply, Tool
from aigent.tools import CALCULATOR_TOOL, execute_tool


def reaching[Answer](llm: Llm, call: Callable[[], Answer]) -> Answer:
    """What `call` returns, sent again when the endpoint does not answer; after
    `SPECIALIST_TRIES` of those, SpecialistUnreachable."""
    dropped: list[Exception] = []
    while len(dropped) < SPECIALIST_TRIES:
        try:
            return call()
        except UNREACHABLE as down:
            dropped.append(down)
    last = dropped[-1]
    raise SpecialistUnreachable(
        f"{llm.client} did not answer in {SPECIALIST_TRIES} tries: {type(last).__name__}: {last}"
    ) from last


def warm_up(small: Llm) -> None:
    """One token from the endpoint, so its cold start is waited out before the first job."""
    ready = Request("warm-up", [{"role": "user", "content": "ready"}], 1)
    reaching(small, lambda: small.create(ready))


def _told(
    usd: float,
    text: str,
    replies: Sequence[Reply],
    calls: Sequence[str] = (),
    read: Sequence[Passage] = (),
    conversation: Sequence[Msg] = (),
) -> Told:
    """A job's reply with what its own turns cost, `usd`, and the tokens of every turn it took."""
    return Told(
        text,
        usd,
        sum(reply.usage.input_tokens for reply in replies),
        sum(reply.usage.output_tokens for reply in replies),
        tuple(calls),
        tuple(read),
        tuple(conversation),
    )


def report(year: int) -> str:
    """How the prompt names the report for the financial year it ends: 2025 is 2024-25."""
    return f"Annual report {year - 1}-{year % 100:02d}"


def _answered(
    reply: Reply, job: FigureJob, search: Dispatch, calls: list[str], read: list[Passage]
) -> Msg:
    """A turn's tool calls, run, as one message of results: a search of the job's one report,
    whatever scope the model asks for, and calculate; any other tool is refused."""
    answered: list[Block] = []
    for block in reply.blocks:
        if block.get("type") != "tool_use":
            continue
        name, arguments = str(block["name"]), cast(dict[str, object], block.get("input") or {})
        if name == REPORT_SEARCH_TOOL["name"]:
            scoped = {"query": str(arguments.get("query", "")), "company": job.ticker}
            content, is_error = search(SEARCH, {**scoped, "year": str(job.year)})
            read.extend(passages_in(content))
        elif name == CALCULATOR_TOOL["name"]:
            calls.append(str(arguments.get("expression", "")))
            content, is_error = execute_tool(name, arguments)
        else:
            content, is_error = f"Error: {name} is not available here", True
        answered.append(
            {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": content,
                "is_error": is_error,
            }
        )
    return {"role": "user", "content": answered}


def opening(job: FigureJob, names: Mapping[str, str] = COMPANIES) -> list[Msg]:
    """A job's first user turn: its question, the company `names` gives its ticker, its report."""
    _, user = research_messages(job.question, names[job.ticker], report(job.year))
    return [{"role": "user", "content": user}]


TOOLS: tuple[Tool, ...] = (REPORT_SEARCH_TOOL, CALCULATOR_TOOL)


def turn_request(
    messages: Sequence[Msg],
    cap: int,
    model: str | None = None,
    cache: Cache | None = None,
    system: str | None = None,
    tools: Sequence[Tool] = TOOLS,
) -> Request:
    """One turn of the report analyst's loop: the trained prompt, the search and calculate, unless
    a teacher's `system` and `tools` are given."""
    trained, _ = research_messages("", "", "")
    return Request(
        "extract_figures",
        messages,
        cap,
        system=system or trained,
        model=model,
        tools=list(tools),
        cache_control=cache,
    )


def research(
    llm: Llm,
    search: Dispatch,
    model: str | None = None,
    names: Mapping[str, str] = COMPANIES,
    cache: Cache | None = None,
    system: str | None = None,
    tools: Sequence[Tool] = TOOLS,
) -> Callable[[FigureJob], Told]:
    """The report analyst's loop on `llm`, asked as `model`: a job's question about the company
    `names` gives its ticker, searched through `search` and calculated, every turn of it kept in
    `Told.conversation`; `cache` marks each turn for a prompt cache, where the wire has one, and a
    teacher's `system` and `tools` replace the trained ones."""
    cap = max_tokens("RESEARCH_TURN", spec(llm.client))

    def answer(job: FigureJob) -> Told:
        usd = 0.0  # this job's turns alone: another thread may be billing the same llm
        messages = opening(job, names)
        replies: list[Reply] = []
        calls: list[str] = []
        read: list[Passage] = []
        for _ in range(SPECIALIST_TOOL_TURNS):
            request = turn_request(messages, cap, model, cache, system, tools)
            reply = llm.turn(request)
            usd += llm.price(request, reply.usage)
            replies.append(reply)
            asked: Msg = {"role": "assistant", "content": reply.blocks}
            if reply.stop_reason != "tool_use":
                return _told(usd, reply.text, replies, calls, read, [*messages, asked])
            messages = [*messages, asked, _answered(reply, job, search, calls, read)]
        return _told(usd, "", replies, calls, read, messages)  # still working: no answer

    return answer


def report_analyst(
    small: Llm, big: Llm, search: Dispatch, model: str | None = GENERALIST_MODEL
) -> Specialist[FigureJob, Found]:
    """`extract_figures`: the figures a question needs from one report, found through `search`
    (the served `search_filings`), on the prompt and tools the adapter is trained on; `small`
    serves the adapter, `big` is the generalist, asked as `model`."""
    served = research(small, search)
    return Specialist(
        "extract_figures",
        lambda job: reaching(small, lambda: served(job)),
        research(big, search, model),
        check_figures,
    )
