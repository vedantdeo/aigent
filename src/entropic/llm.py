"""The one module that talks to a model: every request is counted, admitted, sent, billed, traced.

`config` builds the client and `pricing` knows the prices; this is where they meet a request.
Counting is free and comes first, so a call whose worst case would carry its budget past the
ceiling is refused before it is sent — and a concurrent batch is admitted as a whole.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import anthropic
from anthropic import Omit, omit
from anthropic.lib.streaming import MessageStream
from anthropic.types import (
    Message,
    MessageParam,
    OutputConfigParam,
    ParsedMessage,
    TextBlockParam,
    ThinkingConfigParam,
    ToolParam,
    Usage,
)
from pydantic import BaseModel, ValidationError

from entropic.config import MAX_PARALLEL_CALLS, MODEL, get_client
from entropic.pricing import Budget, assert_request_within_budget


@dataclass(frozen=True)
class Request:
    """One call, exactly as it will be sent. `step` names it in the trace and is never sent."""

    step: str
    messages: Sequence[MessageParam]
    max_tokens: int
    system: str | Sequence[TextBlockParam] | Omit = omit
    model: str = MODEL
    tools: Sequence[ToolParam] | Omit = omit
    thinking: ThinkingConfigParam | Omit = omit
    output_config: OutputConfigParam | Omit = omit

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
    usage: Usage
    usd: float


class StepFailed(RuntimeError):
    """A call came back refused, truncated or unparseable. It was billed all the same."""


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

    def _admit(self, requests: Sequence[Request], schema: type[BaseModel] | None = None) -> None:
        # The sum, not each call: calls in flight cannot be recalled, so a batch that cannot all
        # fit must not start.
        worst = 0.0
        for request in requests:
            tokens = self.count(request, schema)
            cost = assert_request_within_budget(request.model, tokens, request.max_tokens)
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
        usage = Usage(input_tokens=self.count(request, schema), output_tokens=request.max_tokens)
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

    def _bill(self, request: Request, usage: Usage) -> None:
        with self._lock:
            usd = self.budget.charge(request.model, usage)
            self.trace.append(Call(request.step, request.model, usage, usd))


def describe(trace: Sequence[Call]) -> str:
    """The trace as a table: one line per call, then the total."""
    lines = [f"  {'step':<22}{'model':<18}{'in':>7}{'out':>6}{'usd':>10}"]
    for call in trace:
        lines.append(
            f"  {call.step:<22}{call.model:<18}{call.usage.input_tokens:>7}"
            f"{call.usage.output_tokens:>6}{call.usd:>10.5f}"
        )
    calls = f"{len(trace)} calls"
    lines.append(f"  {calls:<53}{sum(call.usd for call in trace):>10.5f}")
    return "\n".join(lines)
