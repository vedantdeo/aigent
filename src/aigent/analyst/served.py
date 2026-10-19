"""The report analyst wired to models: each job on the prompt its fine-tune is trained on, with a
search of one report and calculate, put to the served adapter and, as its fallback, to the
generalist; and the request that warms the endpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from aigent.llm.messages import Block, Msg, Reply
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
    llm: Llm,
    before: float,
    text: str,
    replies: Sequence[Reply],
    calls: Sequence[str] = (),
    read: Sequence[Passage] = (),
) -> Told:
    """A job's reply with what `llm` billed for it since `before`, over every turn it took."""
    return Told(
        text,
        llm.spent_usd - before,
        sum(reply.usage.input_tokens for reply in replies),
        sum(reply.usage.output_tokens for reply in replies),
        tuple(calls),
        tuple(read),
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


def _research(llm: Llm, search: Dispatch, model: str | None = None) -> Callable[[FigureJob], Told]:
    cap = max_tokens("FIGURE_ANSWER", spec(llm.client))

    def research(job: FigureJob) -> Told:
        before = llm.spent_usd
        system, user = research_messages(job.question, COMPANIES[job.ticker], report(job.year))
        messages: list[Msg] = [{"role": "user", "content": user}]
        replies: list[Reply] = []
        calls: list[str] = []
        read: list[Passage] = []
        for _ in range(SPECIALIST_TOOL_TURNS):
            request = Request(
                "extract_figures",
                messages,
                cap,
                system=system,
                model=model,
                tools=[REPORT_SEARCH_TOOL, CALCULATOR_TOOL],
            )
            reply = llm.turn(request)
            replies.append(reply)
            if reply.stop_reason != "tool_use":
                return _told(llm, before, reply.text, replies, calls, read)
            asked: Msg = {"role": "assistant", "content": reply.blocks}
            messages = [*messages, asked, _answered(reply, job, search, calls, read)]
        return _told(llm, before, "", replies, calls, read)  # still working: nothing to check

    return research


def report_analyst(
    small: Llm, big: Llm, search: Dispatch, model: str | None = GENERALIST_MODEL
) -> Specialist[FigureJob, Found]:
    """`extract_figures`: the figures a question needs from one report, found through `search`
    (the served `search_filings`), on the prompt and tools the adapter is trained on; `small`
    serves the adapter, `big` is the generalist, asked as `model`."""
    served = _research(small, search)
    return Specialist(
        "extract_figures",
        lambda job: reaching(small, lambda: served(job)),
        _research(big, search, model),
        check_figures,
    )
