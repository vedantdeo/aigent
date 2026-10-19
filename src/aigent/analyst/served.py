"""The report analyst wired to models: each job on the prompt its fine-tune was trained on, put to
the served adapter and, as its fallback, to the generalist; and the request that warms the endpoint.
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
from aigent.analyst.specialists import FigureJob, Specialist, Told, check_figures
from aigent.errors import SpecialistUnreachable
from aigent.extraction.figures import Figures, figure_messages
from aigent.llm.adapters import spec
from aigent.llm.adapters.openai import UNREACHABLE
from aigent.llm.core import Llm, Request
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
    llm: Llm, before: float, text: str, replies: Sequence[Reply], calls: Sequence[str] = ()
) -> Told:
    """A job's reply with what `llm` billed for it since `before`, over every turn it took."""
    return Told(
        text,
        llm.spent_usd - before,
        sum(reply.usage.input_tokens for reply in replies),
        sum(reply.usage.output_tokens for reply in replies),
        tuple(calls),
    )


def _calculated(reply: Reply, calls: list[str]) -> Msg:
    """A turn's calculate calls, run, as one message of results; any other tool is refused."""
    answered: list[Block] = []
    for block in reply.blocks:
        if block.get("type") != "tool_use":
            continue
        name, arguments = str(block["name"]), cast(dict[str, object], block.get("input") or {})
        if name == CALCULATOR_TOOL["name"]:
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


def _extract(llm: Llm, model: str | None = None) -> Callable[[FigureJob], Told]:
    cap = max_tokens("FIGURE_ANSWER", spec(llm.client))

    def extract(job: FigureJob) -> Told:
        before = llm.spent_usd
        system, user = figure_messages(job.passage, job.source, job.wanted)
        messages: list[Msg] = [{"role": "user", "content": user}]
        replies: list[Reply] = []
        calls: list[str] = []
        for _ in range(SPECIALIST_TOOL_TURNS):
            request = Request(
                "extract_figures",
                messages,
                cap,
                system=system,
                model=model,
                tools=[CALCULATOR_TOOL],
            )
            reply = llm.turn(request)
            replies.append(reply)
            if reply.stop_reason != "tool_use":
                return _told(llm, before, reply.text, replies, calls)
            asked: Msg = {"role": "assistant", "content": reply.blocks}
            messages = [*messages, asked, _calculated(reply, calls)]
        return _told(llm, before, "", replies, calls)  # still calculating: nothing to check

    return extract


def report_analyst(
    small: Llm, big: Llm, model: str | None = GENERALIST_MODEL
) -> Specialist[FigureJob, Figures]:
    """`extract_figures`: a passage's figures, on the prompt and the calculate tool the adapter was
    trained on; `small` serves the adapter, `big` is the generalist, asked as `model`."""
    served = _extract(small)
    return Specialist(
        "extract_figures",
        lambda job: reaching(small, lambda: served(job)),
        _extract(big, model),
        check_figures,
    )
