"""The one module that talks to a model: every request is counted, admitted, sent, billed, traced.

`config` builds the client and `pricing` knows the prices; this is where they meet a request.
Counting is free and comes first, so a call whose worst case would carry its budget past the
ceiling is refused before it is sent — and a concurrent batch is admitted as a whole.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TypeGuard, cast

import anthropic
from anthropic import Omit, omit
from anthropic.lib.streaming import MessageStream
from anthropic.lib.tools import BetaFunctionTool, ToolError, beta_tool
from anthropic.types import (
    CacheControlEphemeralParam,
    Message,
    MessageParam,
    OutputConfigParam,
    ParsedMessage,
    TextBlockParam,
    ThinkingConfigParam,
    ToolParam,
    ToolUnionParam,
    Usage,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaMessage,
    BetaMessageParam,
    BetaOutputConfigParam,
    BetaTextBlockParam,
    BetaThinkingConfigParam,
    BetaToolUnionParam,
)
from anthropic.types.tool_param import InputSchema
from pydantic import BaseModel, ValidationError

from entropic.config import (
    MAX_AGENT_TURNS,
    MAX_PARALLEL_CALLS,
    MAX_USD_PER_REQUEST,
    MAX_USD_PER_TURN,
    MIN_TOKENS_FINAL_ANSWER,
    MODEL,
    get_client,
)
from entropic.errors import BudgetExceeded, StepFailed, TurnsExhausted
from entropic.pricing import (
    Budget,
    LlmUsage,
    affordable_output_tokens,
    assert_request_within_budget,
    web_searches,
)


@dataclass(frozen=True)
class Request:
    """One call, exactly as it will be sent. `step` names it in the trace and is never sent."""

    step: str
    messages: Sequence[MessageParam]
    max_tokens: int
    system: str | Sequence[TextBlockParam] | Omit = omit
    model: str = MODEL
    tools: Sequence[ToolUnionParam] | Omit = omit
    thinking: ThinkingConfigParam | Omit = omit
    output_config: OutputConfigParam | Omit = omit
    cache_control: CacheControlEphemeralParam | Omit = omit

    @classmethod
    def ask(
        cls,
        step: str,
        system: str | Sequence[TextBlockParam],
        prompt: str,
        max_tokens: int,
        *,
        model: str = MODEL,
        thinking: ThinkingConfigParam | Omit = omit,
    ) -> Request:
        """A single-turn request: one system prompt, one user message."""
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        return cls(step, messages, max_tokens, system=system, model=model, thinking=thinking)


@dataclass(frozen=True)
class Call:
    """One line of the trace."""

    step: str
    model: str
    usage: LlmUsage
    usd: float


@dataclass(frozen=True)
class ToolRun:
    """How a tool-using conversation ended: its last message, why it was told to answer before it
    was done, if it was, and every turn it took, the last included."""

    message: BetaMessage
    cut_short: str | None = None
    turns: tuple[BetaMessage, ...] = ()


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

    The budget defaults to one run's ceiling; pass one to share it or scope it. With `rehearse`,
    the first call is counted and priced, and nothing is sent.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        budget: Budget | None = None,
        rehearse: bool = False,
    ) -> None:
        self._client = client
        self.budget = budget if budget is not None else Budget()
        self.trace: list[Call] = []
        self.rehearse = rehearse
        self._lock = threading.Lock()

    @classmethod
    def for_eval(cls, client: anthropic.Anthropic | None = None) -> Llm:
        """An `Llm` for an eval's task or grader: every call counted and checked against the
        per-request ceiling, but no run ceiling of its own — the runner admits and bills the run."""
        return cls(client, budget=Budget(limit_usd=math.inf, scope="eval"))

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = get_client()
        return self._client

    @property
    def spent_usd(self) -> float:
        return self.budget.spent_usd

    def count(self, request: Request, schema: type[BaseModel] | None = None) -> int:
        """The input tokens `request` would send, schema included. Free: nothing is generated."""
        output_format: type | Omit = omit if schema is None else schema
        return self.client.messages.count_tokens(
            model=request.model,
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            thinking=request.thinking,
            output_config=request.output_config,
            output_format=output_format,
            cache_control=request.cache_control,
        ).input_tokens

    def create(self, request: Request) -> Message:
        """Send one request; the whole message, for callers that read its blocks."""
        self._admit([request])
        return self._create(request)

    def text(self, request: Request) -> str:
        """Send one request; its text. Raises `StepFailed` on a refusal or a truncation."""
        self._admit([request])
        return self._text(request)

    def parse[Record: BaseModel](
        self, request: Request, schema: type[Record]
    ) -> ParsedMessage[Record]:
        """Send one request for a `schema` record; the whole message, whose `parsed_output` is
        None when the output did not validate."""
        self._admit([request], schema)
        return self._parse(request, schema)

    def record[Record: BaseModel](self, request: Request, schema: type[Record]) -> Record:
        """Send one request for a `schema` record. Raises `StepFailed` when there is none."""
        self._admit([request], schema)
        return self._record(request, schema)

    @contextmanager
    def stream(self, request: Request) -> Iterator[MessageStream]:
        """Send one request as a stream; billed from the final message once the stream closes."""
        self._admit([request])
        with self.client.messages.stream(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            thinking=request.thinking,
            output_config=request.output_config,
            cache_control=request.cache_control,
        ) as stream:
            yield stream
            final = stream.get_final_message()
        self._bill(request, final.usage)

    def gather_text(self, requests: Sequence[Request]) -> list[str]:
        """`text` for every request concurrently, in request order, once all are admitted."""
        self._admit(requests)
        return self._fan_out(requests, self._text)

    def gather_records[Record: BaseModel](
        self, requests: Sequence[Request], schema: type[Record]
    ) -> list[Record]:
        """`record` for every request concurrently, in request order, once all are admitted."""
        self._admit(requests, schema)
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
        if isinstance(request.tools, Omit) or not request.tools:
            raise ValueError(f"{request.step}: run_tools needs at least one tool")
        messages: list[MessageParam] = list(request.messages)
        first = replace(request, step=f"{request.step}:1", messages=messages)
        self._admit([first], limit_usd=MAX_USD_PER_TURN, scope="turn")
        runnable = [_runnable(tool, dispatch) for tool in request.tools if _is_client(tool)]
        server = [cast(BetaToolUnionParam, t) for t in request.tools if not _is_client(t)]
        # What the runner puts on the wire: its own tools first, then the ones the API runs.
        wire = [*(tool.to_dict() for tool in runnable), *server]
        runner = self.client.beta.messages.tool_runner(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=cast(list[BetaMessageParam], messages),
            system=cast("str | list[BetaTextBlockParam] | Omit", request.system),
            thinking=cast("BetaThinkingConfigParam | Omit", request.thinking),
            output_config=cast("BetaOutputConfigParam | Omit", request.output_config),
            cache_control=cast("BetaCacheControlEphemeralParam | Omit", request.cache_control),
            tools=[*runnable, *server],
            max_iterations=max_turns,
        )
        turns: list[BetaMessage] = []
        for turn, message in enumerate(runner, start=1):
            self._bill(replace(request, step=f"{request.step}:{turn}"), message.usage)
            turns.append(message)
            if message.stop_reason not in ("tool_use", "pause_turn"):
                return ToolRun(message, turns=tuple(turns))
            if turn == max_turns:
                break
            messages = [
                *messages,
                cast(MessageParam, {"role": "assistant", "content": message.content}),
            ]
            if message.stop_reason == "tool_use":
                # Runs the tools now; the runner reuses these results rather than rerun them.
                results = runner.generate_tool_call_response()
                if results is None:
                    return ToolRun(message, turns=tuple(turns))
                messages.append(cast(MessageParam, results))
            upcoming = replace(request, step=f"{request.step}:{turn + 1}", messages=messages)
            stop: Exception | None
            if finish is not None and turn + 1 == max_turns:
                stop = TurnsExhausted(
                    f"{request.step}: the last of {max_turns} turns is kept for an answer"
                )
            else:
                stop = self._refusal(upcoming)
            if stop is None:
                continue
            if finish is None or message.stop_reason != "tool_use":
                raise stop
            answer = self._last_answer(upcoming, wire, finish, stop)
            return ToolRun(answer, str(stop), (*turns, answer))
        raise TurnsExhausted(f"{request.step}: still asking for tools after {max_turns} turns")

    def _refusal(self, upcoming: Request) -> BudgetExceeded | None:
        """Why `upcoming` may not be sent as another tool turn, or None once it is admitted."""
        try:
            self._admit([upcoming], limit_usd=MAX_USD_PER_TURN, scope="turn")
        except BudgetExceeded as refused:
            return refused
        return None

    def _last_answer(
        self,
        upcoming: Request,
        wire: Sequence[BetaToolUnionParam],
        finish: str,
        stop: Exception,
    ) -> BetaMessage:
        """Send `upcoming` as a last answer instead: `finish` after its tool results, and as much
        output as the per-turn ceiling and the budget allow. Raises `stop` if that is too little,
        or if the model asks for tools anyway."""
        *history, results = upcoming.messages
        blocks = [*cast(list[object], results["content"]), {"type": "text", "text": finish}]
        told = cast(MessageParam, {"role": "user", "content": blocks})
        answer = replace(upcoming, step=f"{upcoming.step} answer", messages=[*history, told])
        room = min(MAX_USD_PER_TURN, self.budget.limit_usd - self.budget.spent_usd)
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
        self._admit([answer], limit_usd=MAX_USD_PER_TURN, scope="turn")
        # Sent as the runner sends a turn, bar the messages and the cap, so the cache still holds.
        message = self.client.beta.messages.parse(
            model=answer.model,
            max_tokens=answer.max_tokens,
            messages=cast(list[BetaMessageParam], answer.messages),
            system=cast("str | list[BetaTextBlockParam] | Omit", answer.system),
            thinking=cast("BetaThinkingConfigParam | Omit", answer.thinking),
            output_config=cast("BetaOutputConfigParam | Omit", answer.output_config),
            cache_control=cast("BetaCacheControlEphemeralParam | Omit", answer.cache_control),
            tools=wire,
        )
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
    ) -> None:
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
            self.budget.admit(worst)

    def _fan_out[Result](
        self, requests: Sequence[Request], send: Callable[[Request], Result]
    ) -> list[Result]:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
            return list(pool.map(send, requests))

    def _create(self, request: Request) -> Message:
        response = self.client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            thinking=request.thinking,
            output_config=request.output_config,
            cache_control=request.cache_control,
        )
        self._bill(request, response.usage)
        return response

    def _text(self, request: Request) -> str:
        response = self._create(request)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise StepFailed(f"{request.step}: stop_reason={response.stop_reason}")
        return "".join(block.text for block in response.content if block.type == "text")

    def _parse[Record: BaseModel](
        self, request: Request, schema: type[Record]
    ) -> ParsedMessage[Record]:
        try:
            response = self.client.messages.parse(
                model=request.model,
                max_tokens=request.max_tokens,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                thinking=request.thinking,
                output_config=request.output_config,
                output_format=schema,
                # The SDK's parse has no cache_control parameter; the API takes it all the same.
                extra_body=_cache_body(request),
            )
        except ValidationError:
            # The SDK validates as it reads, so a cut-off record raises before its usage is seen.
            response = self._unreadable(request, schema)
        self._bill(request, response.usage)
        return response

    def _unreadable[Record: BaseModel](
        self, request: Request, schema: type[Record]
    ) -> ParsedMessage[Record]:
        """A reply the SDK could not read, as an empty message billed at the call's worst case:
        exact for a record cut off at `max_tokens`, which is what makes one unreadable."""
        tokens = self.count(request, schema)
        if _writes_cache(request):
            usage = Usage(
                input_tokens=0, cache_creation_input_tokens=tokens, output_tokens=request.max_tokens
            )
        else:
            usage = Usage(input_tokens=tokens, output_tokens=request.max_tokens)
        return ParsedMessage[Record](
            id="unreadable",
            type="message",
            role="assistant",
            model=request.model,
            content=[],
            stop_reason="max_tokens",
            stop_sequence=None,
            usage=usage,
        )

    def _record[Record: BaseModel](self, request: Request, schema: type[Record]) -> Record:
        response = self._parse(request, schema)
        if response.parsed_output is None:
            raise StepFailed(f"{request.step}: nothing parsed, stop_reason={response.stop_reason}")
        return response.parsed_output

    def _bill(self, request: Request, usage: LlmUsage) -> None:
        with self._lock:
            usd = self.budget.charge(request.model, usage)
            self.trace.append(Call(request.step, request.model, usage, usd))


def _cache_markers(request: Request) -> list[CacheControlEphemeralParam]:
    """Every cache marker a request carries: the top-level one and any on its system blocks."""
    markers = [] if isinstance(request.cache_control, Omit) else [request.cache_control]
    for block in [] if isinstance(request.system, (str, Omit)) else request.system:
        marker = block.get("cache_control")
        if marker is not None:
            markers.append(marker)
    return markers


def _writes_cache(request: Request) -> bool:
    """Whether a request can write to the cache: automatic caching, or a marked system block."""
    return bool(_cache_markers(request))


def _cache_body(request: Request) -> dict[str, object] | None:
    return (
        None
        if isinstance(request.cache_control, Omit)
        else {"cache_control": request.cache_control}
    )


def _is_client(tool: ToolUnionParam) -> TypeGuard[ToolParam]:
    """A tool we run ourselves, as opposed to one the API runs or defines."""
    return tool.get("type") in (None, "custom")


def _web_searches(request: Request) -> int:
    """The most web searches a request can run. Refuses a tool type pricing does not model, and a
    web search with no `max_uses`: either would leave the worst case unbounded."""
    searches = 0
    for tool in [] if isinstance(request.tools, Omit) else request.tools:
        if _is_client(tool):
            continue
        kind = str(tool.get("type"))
        if not kind.startswith("web_search_"):
            raise BudgetExceeded(
                f"{request.step}: pricing does not model {kind}, so no ceiling can hold it."
            )
        uses = cast(Mapping[str, object], tool).get("max_uses")
        if not isinstance(uses, int):
            raise BudgetExceeded(
                f"{request.step}: a web search with no max_uses could search without limit, so no "
                "ceiling can hold it. Set max_uses."
            )
        searches += uses
    return searches


def _runnable(tool: ToolParam, dispatch: Dispatch) -> BetaFunctionTool[Callable[..., str]]:
    """A tool the runner can call, sent exactly as `tool` is written and run through `dispatch`."""
    name = tool["name"]

    def run(**arguments: object) -> str:
        content, is_error = dispatch(name, arguments)
        if is_error:
            raise ToolError(content)
        return content

    return beta_tool(
        run,
        name=name,
        description=tool.get("description"),
        input_schema=cast(InputSchema, tool["input_schema"]),
        strict=tool.get("strict"),
    )


def describe(trace: Sequence[Call]) -> str:
    """The trace as a table, one line per call, then the total. `in` is every input token, cached
    or not; `cached` is how many of them were read from the cache."""
    lines = [f"  {'step':<22}{'model':<18}{'in':>7}{'cached':>8}{'out':>6}{'web':>5}{'usd':>10}"]
    for call in trace:
        usage = call.usage
        read = usage.cache_read_input_tokens or 0
        total = usage.input_tokens + (usage.cache_creation_input_tokens or 0) + read
        lines.append(
            f"  {call.step:<22}{call.model:<18}{total:>7}{read:>8}"
            f"{usage.output_tokens:>6}{web_searches(usage):>5}{call.usd:>10.5f}"
        )
    calls = f"{len(trace)} calls"
    lines.append(f"  {calls:<66}{sum(call.usd for call in trace):>10.5f}")
    return "\n".join(lines)
