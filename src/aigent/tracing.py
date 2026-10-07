"""Tracing: each eval row, model call, tool call and grade as one nested tree in Langfuse.

Silent unless both Langfuse keys are set and `config.TRACING` is not `off`, so tests, dry runs and a
machine without an account trace nothing. The only module that imports the Langfuse SDK.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Literal, Protocol, cast

from aigent import guardrails
from aigent.config import TRACE_ENVIRONMENT, TRACING
from aigent.llm.messages import Usage

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse._client.span import LangfuseObservationWrapper

Kind = Literal["span", "agent", "generation", "tool", "retriever", "evaluator"]

# Runs one tool call, `(name, input)` in and `(content, is_error)` out, as `llm.Dispatch` does.
ToolCall = Callable[[str, dict[str, object]], tuple[str, bool]]


class Observation(Protocol):
    """One open node of the tree, filled in as its work finishes."""

    def finish(
        self,
        output: object = None,
        *,
        error: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None: ...

    def bill(self, model: str, usage: Usage, usd: float) -> None: ...

    def score(self, name: str, passed: bool, comment: str = "") -> None: ...


class Tracer(Protocol):
    """Opens observations nested under whichever one is open in this context."""

    def observe(
        self,
        name: str,
        kind: Kind = "span",
        *,
        inputs: object = None,
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] = (),
        session: str | None = None,
    ) -> AbstractContextManager[Observation]: ...

    def flush(self) -> None: ...


class _Unseen:
    def finish(
        self,
        output: object = None,
        *,
        error: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        pass

    def bill(self, model: str, usage: Usage, usd: float) -> None:
        pass

    def score(self, name: str, passed: bool, comment: str = "") -> None:
        pass


class Silent:
    """The tracer when tracing is off: every observation is a no-op."""

    def observe(
        self,
        name: str,
        kind: Kind = "span",
        *,
        inputs: object = None,
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] = (),
        session: str | None = None,
    ) -> AbstractContextManager[Observation]:
        return nullcontext(_Unseen())

    def flush(self) -> None:
        pass


class _Observed:
    def __init__(self, span: LangfuseObservationWrapper) -> None:
        self._span = span

    def finish(
        self,
        output: object = None,
        *,
        error: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._span.update(
            output=output,
            metadata=None if metadata is None else dict(metadata),
            level="ERROR" if error is not None else None,
            status_message=error,
        )

    def bill(self, model: str, usage: Usage, usd: float) -> None:
        self._span.update(
            model=model,
            usage_details={
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_tokens,
                "cache_creation_input_tokens": usage.cache_write_tokens,
            },
            cost_details={
                "total": usd
            },  # `pricing`'s figure, web searches and local models included
            metadata={"web_searches": usage.web_searches} if usage.web_searches else None,
        )

    def score(self, name: str, passed: bool, comment: str = "") -> None:
        from langfuse.api import ScoreDataType

        self._span.score_trace(
            name=name, value=float(passed), data_type=ScoreDataType.BOOLEAN, comment=comment
        )


class LangfuseTracer:
    """Observations sent to Langfuse, in the background, by its OpenTelemetry exporter."""

    def __init__(self, client: Langfuse | None = None) -> None:
        from langfuse import Langfuse

        self._client = (
            client if client is not None else Langfuse(environment=TRACE_ENVIRONMENT, mask=masked)
        )

    @contextmanager
    def observe(
        self,
        name: str,
        kind: Kind = "span",
        *,
        inputs: object = None,
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] = (),
        session: str | None = None,
    ) -> Iterator[Observation]:
        from langfuse import propagate_attributes

        opened = self._client.start_as_current_observation(
            name=name,
            as_type=kind,
            input=inputs,
            metadata=None if metadata is None else dict(metadata),
        )
        with opened as span:
            attributes = (
                propagate_attributes(tags=list(tags) or None, session_id=session)
                if tags or session
                else nullcontext()
            )
            with attributes:
                observed = _Observed(cast("LangfuseObservationWrapper", span))
                try:
                    yield observed
                except Exception as exc:
                    observed.finish(error=f"{type(exc).__name__}: {exc}")
                    raise

    def flush(self) -> None:
        self._client.flush()


def masked(*, data: object, **_: object) -> object:
    """Langfuse's mask: what any observation sends, with credentials redacted before it leaves."""
    return guardrails.redact(data)


_active: Tracer | None = None


def enabled() -> bool:
    """Whether tracing would send anything: not switched off, and both keys present."""
    keys = os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    return TRACING != "off" and bool(keys)


def tracer() -> Tracer:
    """The tracer this process traces with, chosen on first use."""
    global _active
    if _active is None:
        _active = LangfuseTracer() if enabled() else Silent()
    return _active


def use(chosen: Tracer | None) -> None:
    """Trace with `chosen` from now on; None chooses again on next use. For tests."""
    global _active
    _active = chosen


def traced_tools(dispatch: ToolCall, kinds: Mapping[str, Kind] | None = None) -> ToolCall:
    """`dispatch` with each call recorded as a `tool` observation, or the kind `kinds` names."""

    def run(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        kind = (kinds or {}).get(name, "tool")
        with tracer().observe(name, kind, inputs=arguments) as seen:
            content, is_error = dispatch(name, arguments)
            seen.finish(content, error=content if is_error else None)
        return content, is_error

    return run
