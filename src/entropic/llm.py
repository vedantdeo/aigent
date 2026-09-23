"""The one module that talks to a model: every request is counted, admitted, sent, billed, traced.

`adapters` own the wire formats, `config` builds the client and `pricing` knows the prices; this is
where they meet a request. Counting is free and comes first, so a call whose worst case would carry
its budget past the ceiling is refused before it is sent — and a concurrent batch is admitted whole.

The tool-runner and streaming paths still speak Anthropic here rather than through an adapter: both
are Anthropic-only features, and generalising them with one implementation in hand would be a
guess. They move the day a second wire supports them.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import cast

from pydantic import BaseModel

from entropic.adapters import Streamed, ToolSession, build
from entropic.config import (
    CLIENT,
    MAX_AGENT_TURNS,
    MAX_PARALLEL_CALLS,
    MAX_USD_PER_REQUEST,
    MAX_USD_PER_TURN,
    MIN_TOKENS_FINAL_ANSWER,
    MODEL,
)
from entropic.errors import BudgetExceeded, StepFailed, TurnsExhausted, Unreadable
from entropic.messages import (
    Block,
    Cache,
    Msg,
    OutputConfig,
    Parsed,
    Reply,
    Thinking,
    Tool,
    Usage,
    is_client_tool,
)
from entropic.pricing import Budget, affordable_output_tokens, assert_request_within_budget


@dataclass(frozen=True)
class Request:
    """One call, exactly as it will be sent. `step` names it in the trace and is never sent."""

    step: str
    messages: Sequence[Msg]
    max_tokens: int
    system: str | Sequence[Block] | None = None
    model: str = MODEL
    tools: Sequence[Tool] | None = None
    thinking: Thinking | None = None
    output_config: OutputConfig | None = None
    cache_control: Cache | None = None

    @classmethod
    def ask(
        cls,
        step: str,
        system: str | Sequence[Block],
        prompt: str,
        max_tokens: int,
        *,
        model: str = MODEL,
        thinking: Thinking | None = None,
    ) -> Request:
        """A single-turn request: one system prompt, one user message."""
        messages: list[Msg] = [{"role": "user", "content": prompt}]
        return cls(step, messages, max_tokens, system=system, model=model, thinking=thinking)


@dataclass(frozen=True)
class Call:
    """One line of the trace."""

    step: str
    model: str
    usage: Usage
    usd: float


@dataclass(frozen=True)
class ToolRun:
    """How a tool-using conversation ended: its last message, why it was told to answer before it
    was done, if it was, and every turn it took, the last included."""

    message: Reply
    cut_short: str | None = None
    turns: tuple[Reply, ...] = ()


# Runs one tool call: `(name, input)` in, `(content, is_error)` out, as `tools.execute_tool` does.
Dispatch = Callable[[str, dict[str, object]], tuple[str, bool]]


class Rehearsed(Exception):
    """Raised on a dry run in place of the first call, carrying what that call would cost."""

    def __init__(self, request: Request, input_tokens: int, worst_usd: float) -> None:
        super().__init__(f"{request.step}: {input_tokens} input tokens, up to ${worst_usd:.4f}")
        self.request = request
        self.input_tokens = input_tokens
        self.worst_usd = worst_usd


class Llm:
    """One client, one budget, one trace.

    `client` names the endpoint to talk to (`adapters.CLIENTS`), which decides the wire, the
    models and any cap it wants different. `sdk` puts a vendor client under that wire instead of
    letting it build one: a test scripts replies with it, and an eval hands its own down so 54
    tasks and a judge share one connection rather than opening one each.

    The budget defaults to one run's ceiling; pass one to share it or scope it. With `rehearse`,
    the first call is counted and priced, and nothing is sent.
    """

    def __init__(
        self,
        client: str = CLIENT,
        *,
        sdk: object | None = None,
        budget: Budget | None = None,
        rehearse: bool = False,
    ) -> None:
        self._client = client
        self._adapter = build(client, sdk)
        self.budget = budget if budget is not None else Budget()
        self.trace: list[Call] = []
        self.rehearse = rehearse
        self._lock = threading.Lock()

    @classmethod
    def for_eval(cls, client: str = CLIENT, *, sdk: object | None = None) -> Llm:
        """An `Llm` for an eval's task or grader: every call counted and checked against the
        per-request ceiling, but no run ceiling of its own — the runner admits and bills the run."""
        return cls(client, sdk=sdk, budget=Budget(limit_usd=math.inf, scope="eval"))

    @property
    def client(self) -> str:
        """Which client this `Llm` talks to."""
        return self._client

    @property
    def sdk(self) -> object:
        """Its vendor client, to hand to another `Llm` so the two share one connection."""
        return self._adapter.client

    @property
    def spent_usd(self) -> float:
        return self.budget.spent_usd

    def count(self, request: Request, schema: type[BaseModel] | None = None) -> int:
        """The input tokens `request` would send, schema included. Free: nothing is generated."""
        return self._adapter.count(request, schema)

    def create(self, request: Request) -> Reply:
        """Send one request; the whole reply, for callers that read its blocks."""
        with self._held([request]):
            return self._create(request)

    def text(self, request: Request) -> str:
        """Send one request; its text. Raises `StepFailed` on a refusal or a truncation."""
        with self._held([request]):
            return self._text(request)

    def parse[Record: BaseModel](self, request: Request, schema: type[Record]) -> Parsed[Record]:
        """Send one request for a `schema` record; the whole reply, whose `parsed` is None when the
        output did not validate."""
        with self._held([request], schema):
            return self._parse(request, schema)

    def record[Record: BaseModel](self, request: Request, schema: type[Record]) -> Record:
        """Send one request for a `schema` record. Raises `StepFailed` when there is none."""
        with self._held([request], schema):
            return self._record(request, schema)

    @contextmanager
    def stream(self, request: Request) -> Iterator[Streamed]:
        """Send one request as a stream; billed from the final reply once the stream closes."""
        with self._held([request]), self._adapter.streamed(request) as streaming:
            yield streaming
            final = streaming.final()
        self._bill(request, final.usage)

    def gather_text(self, requests: Sequence[Request]) -> list[str]:
        """`text` for every request concurrently, in request order, once all are admitted."""
        with self._held(requests):
            return self._fan_out(requests, self._text)

    def gather_records[Record: BaseModel](
        self, requests: Sequence[Request], schema: type[Record]
    ) -> list[Record]:
        """`record` for every request concurrently, in request order, once all are admitted."""
        with self._held(requests, schema):
            return self._fan_out(requests, lambda request: self._record(request, schema))

    def run_tools(
        self,
        request: Request,
        dispatch: Dispatch,
        *,
        max_turns: int = MAX_AGENT_TURNS,
        finish: str | None = None,
    ) -> ToolRun:
        """Run a tool-using conversation on the SDK's tool runner until the model stops asking for
        tools. Each turn is admitted against the per-turn ceiling before it is sent, billed after.

        With `finish`, a conversation about to run out of turns or budget is sent one last turn with
        `finish` after its tool results, to answer from what it has, if enough output still fits.
        """
        if not request.tools:
            raise ValueError(f"{request.step}: run_tools needs at least one tool")
        messages: list[Msg] = list(request.messages)
        first = replace(request, step=f"{request.step}:1", messages=messages)
        held = self._admit([first], limit_usd=MAX_USD_PER_TURN, scope="turn")
        try:
            session = self._adapter.tools(first, dispatch, max_turns)
            turns: list[Reply] = []
            for turn, message in enumerate(session, start=1):
                self._bill(replace(request, step=f"{request.step}:{turn}"), message.usage)
                self._release(held)
                held = 0.0
                turns.append(message)
                if message.stop_reason not in ("tool_use", "pause_turn"):
                    return ToolRun(message, turns=tuple(turns))
                if turn == max_turns:
                    break
                messages = [*messages, Msg(role="assistant", content=message.blocks)]
                if message.stop_reason == "tool_use":
                    # Runs the tools now; the runner reuses these results rather than rerun them.
                    results = session.tool_results()
                    if results is None:
                        return ToolRun(message, turns=tuple(turns))
                    messages.append(results)
                upcoming = replace(request, step=f"{request.step}:{turn + 1}", messages=messages)
                stop: Exception
                if finish is not None and turn + 1 == max_turns:
                    stop = TurnsExhausted(
                        f"{request.step}: the last of {max_turns} turns is kept for an answer"
                    )
                else:
                    admitted = self._admit_turn(upcoming)
                    if not isinstance(admitted, BudgetExceeded):
                        held = admitted
                        continue
                    stop = admitted
                if finish is None or message.stop_reason != "tool_use":
                    raise stop
                answer = self._last_answer(upcoming, session, finish, stop)
                return ToolRun(answer, str(stop), (*turns, answer))
            raise TurnsExhausted(f"{request.step}: still asking for tools after {max_turns} turns")
        finally:
            self._release(held)

    def _admit_turn(self, upcoming: Request) -> float | BudgetExceeded:
        """Admit `upcoming` as the next tool turn and return what it holds, or the refusal."""
        try:
            return self._admit([upcoming], limit_usd=MAX_USD_PER_TURN, scope="turn")
        except BudgetExceeded as refused:
            return refused

    def _last_answer(
        self,
        upcoming: Request,
        session: ToolSession,
        finish: str,
        stop: Exception,
    ) -> Reply:
        """Send `upcoming` as a last answer instead: `finish` after its tool results, and as much
        output as the per-turn ceiling and the budget allow. Raises `stop` if that is too little,
        or if the model asks for tools anyway."""
        *history, results = upcoming.messages
        blocks = [*cast(Sequence[Block], results["content"]), {"type": "text", "text": finish}]
        told = Msg(role="user", content=cast(Sequence[Block], blocks))
        answer = replace(upcoming, step=f"{upcoming.step} answer", messages=[*history, told])
        left = self.budget.limit_usd - self.budget.spent_usd - self.budget.held_usd
        room = min(MAX_USD_PER_TURN, left)
        fits = affordable_output_tokens(
            answer.model,
            self.count(answer),
            room,
            cached=_writes_cache(answer),
            web_searches=_web_searches(answer),
        )
        answer = replace(answer, max_tokens=min(answer.max_tokens, fits))
        if answer.max_tokens < MIN_TOKENS_FINAL_ANSWER:
            raise stop
        with self._held([answer], limit_usd=MAX_USD_PER_TURN, scope="turn"):
            message = session.answer(answer)
        self._bill(answer, message.usage)
        if message.stop_reason == "tool_use":
            raise stop
        return message

    def _admit(
        self,
        requests: Sequence[Request],
        schema: type[BaseModel] | None = None,
        *,
        limit_usd: float = MAX_USD_PER_REQUEST,
        scope: str = "request",
    ) -> float:
        """Admit every request, holding their summed worst case until `_release`; returns it."""
        # The sum, not each call: calls in flight cannot be recalled, so a batch that cannot all
        # fit must not start.
        worst = 0.0
        for request in requests:
            if any(marker.get("ttl") == "1h" for marker in _cache_markers(request)):
                raise BudgetExceeded(
                    f"{request.step}: asks for the 1-hour cache, whose writes pricing does not "
                    "model, so no ceiling can hold it. Use the 5-minute default."
                )
            searches = _web_searches(request)
            tokens = self.count(request, schema)
            cost = assert_request_within_budget(
                request.model,
                tokens,
                request.max_tokens,
                limit_usd,
                cached=_writes_cache(request),
                scope=scope,
                web_searches=searches,
            )
            if self.rehearse:
                raise Rehearsed(request, tokens, cost)
            worst += cost
        with self._lock:
            self.budget.admit(worst, hold=True)
        return worst

    @contextmanager
    def _held(
        self,
        requests: Sequence[Request],
        schema: type[BaseModel] | None = None,
        *,
        limit_usd: float = MAX_USD_PER_REQUEST,
        scope: str = "request",
    ) -> Iterator[None]:
        """Admitted for the length of the block, which sends and bills; let go however it ends."""
        held = self._admit(requests, schema, limit_usd=limit_usd, scope=scope)
        try:
            yield
        finally:
            self._release(held)

    def _release(self, held: float) -> None:
        with self._lock:
            self.budget.release(held)

    def _fan_out[Result](
        self, requests: Sequence[Request], send: Callable[[Request], Result]
    ) -> list[Result]:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
            return list(pool.map(send, requests))

    def _create(self, request: Request) -> Reply:
        reply = self._adapter.send(request)
        self._bill(request, reply.usage)
        return reply

    def _text(self, request: Request) -> str:
        reply = self._create(request)
        if reply.stop_reason in ("refusal", "max_tokens"):
            raise StepFailed(f"{request.step}: stop_reason={reply.stop_reason}")
        return reply.text

    def _parse[Record: BaseModel](self, request: Request, schema: type[Record]) -> Parsed[Record]:
        try:
            parsed = self._adapter.parse(request, schema, _cache_body(request))
        except Unreadable:
            parsed = self._unreadable(request, schema)
        self._bill(request, parsed.usage)
        return parsed

    def _unreadable[Record: BaseModel](
        self, request: Request, schema: type[Record]
    ) -> Parsed[Record]:
        """A reply no adapter could read, as an empty one billed at the call's worst case: exact
        for a record cut off at `max_tokens`, which is what makes one unreadable."""
        tokens = self.count(request, schema)
        usage = (
            Usage(input_tokens=0, cache_write_tokens=tokens, output_tokens=request.max_tokens)
            if _writes_cache(request)
            else Usage(input_tokens=tokens, output_tokens=request.max_tokens)
        )
        return Parsed[Record](
            text="",
            usage=usage,
            stop_reason="max_tokens",
            model=request.model,
            parsed=None,
        )

    def _record[Record: BaseModel](self, request: Request, schema: type[Record]) -> Record:
        parsed = self._parse(request, schema)
        if parsed.parsed is None:
            raise StepFailed(f"{request.step}: nothing parsed, stop_reason={parsed.stop_reason}")
        return parsed.parsed

    def _bill(self, request: Request, usage: Usage) -> None:
        with self._lock:
            usd = self.budget.charge(request.model, usage)
            self.trace.append(Call(request.step, request.model, usage, usd))


def _cache_markers(request: Request) -> list[Cache]:
    """Every cache marker a request carries: the top-level one and any on its system blocks."""
    markers = [] if request.cache_control is None else [request.cache_control]
    blocks = [] if request.system is None or isinstance(request.system, str) else request.system
    for block in blocks:
        marker = block.get("cache_control")
        if marker is not None:
            markers.append(cast(Cache, marker))
    return markers


def _writes_cache(request: Request) -> bool:
    """Whether a request can write to the cache: automatic caching, or a marked system block."""
    return bool(_cache_markers(request))


def _cache_body(request: Request) -> dict[str, object] | None:
    return None if request.cache_control is None else {"cache_control": request.cache_control}


def _web_searches(request: Request) -> int:
    """The most web searches a request can run. Refuses a tool type pricing does not model, and a
    web search with no `max_uses`: either would leave the worst case unbounded."""
    searches = 0
    for tool in request.tools or []:
        if is_client_tool(tool):
            continue
        kind = str(tool.get("type"))
        if not kind.startswith("web_search_"):
            raise BudgetExceeded(
                f"{request.step}: pricing does not model {kind}, so no ceiling can hold it."
            )
        uses = tool.get("max_uses")
        if not isinstance(uses, int):
            raise BudgetExceeded(
                f"{request.step}: a web search with no max_uses could search without limit, so no "
                "ceiling can hold it. Set max_uses."
            )
        searches += uses
    return searches


def describe(trace: Sequence[Call]) -> str:
    """The trace as a table, one line per call, then the total. `in` is every input token, cached
    or not; `cached` is how many of them were read from the cache."""
    lines = [f"  {'step':<22}{'model':<18}{'in':>7}{'cached':>8}{'out':>6}{'web':>5}{'usd':>10}"]
    for call in trace:
        usage = call.usage
        total = usage.input_tokens + usage.cache_write_tokens + usage.cache_read_tokens
        lines.append(
            f"  {call.step:<22}{call.model:<18}{total:>7}{usage.cache_read_tokens:>8}"
            f"{usage.output_tokens:>6}{usage.web_searches:>5}{call.usd:>10.5f}"
        )
    calls = f"{len(trace)} calls"
    lines.append(f"  {calls:<66}{sum(call.usd for call in trace):>10.5f}")
    return "\n".join(lines)
