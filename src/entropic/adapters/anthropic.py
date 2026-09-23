"""The Anthropic wire: neutral request in, neutral reply out.

The only module that builds Anthropic parameters or reads its responses. `llm` holds one `Anthropic`
and calls it for every request; counting, admission, billing, the trace and concurrency stay there.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

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
    ToolUnionParam,
)
from anthropic.types import Usage as WireUsage
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaMessage,
    BetaMessageParam,
    BetaOutputConfigParam,
    BetaTextBlockParam,
    BetaThinkingConfigParam,
    BetaToolUnionParam,
    BetaUsage,
)
from anthropic.types.tool_param import InputSchema
from pydantic import BaseModel, ValidationError

from entropic.adapters.client import Client, model_of
from entropic.errors import Unreadable
from entropic.messages import Block, Msg, Parsed, Reply, Tool, Usage, is_client_tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from entropic.llm import Dispatch, Request

import os
from pathlib import Path

# The SDK client, named here so `llm` can hold an adapter without importing the SDK itself.
AnthropicClient = anthropic.Anthropic

# Every capability this wire supports; `llm` refuses a request asking for one not listed here.
SUPPORTS = frozenset({"cache", "thinking", "effort", "tools", "web_search", "stream", "schema"})


def usage_of(usage: WireUsage | BetaUsage) -> Usage:
    """An Anthropic usage as ours. The tool runner answers on the beta endpoint; both count the
    same things."""
    searches = usage.server_tool_use.web_search_requests if usage.server_tool_use else 0
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        web_searches=searches,
    )


def reply_of(message: Message | BetaMessage) -> Reply:
    """An Anthropic message as ours, keeping both the blocks and the message itself."""
    return Reply(
        text="".join(block.text for block in message.content if block.type == "text"),
        usage=usage_of(message.usage),
        stop_reason=message.stop_reason,
        model=message.model,
        blocks=tuple(cast(Block, block.model_dump()) for block in message.content),
        raw=message,
    )


def as_parsed[Record: BaseModel](message: ParsedMessage[Record]) -> Parsed[Record]:
    """A parsed Anthropic message as ours, with the record it read."""
    reply = reply_of(message)
    return Parsed[Record](
        text=reply.text,
        usage=reply.usage,
        stop_reason=reply.stop_reason,
        model=reply.model,
        blocks=reply.blocks,
        raw=message,
        parsed=message.parsed_output,
    )


def or_omit[Value](value: Value | None) -> Value | Omit:
    """None means "do not send it", which is what the SDK's `omit` sentinel says."""
    return omit if value is None else value


def system_param(request: Request) -> str | Sequence[TextBlockParam] | Omit:
    """The system prompt as the SDK takes it: a string, blocks, or nothing."""
    if request.system is None or isinstance(request.system, str):
        return or_omit(request.system)
    return cast(Sequence[TextBlockParam], request.system)


def _runnable(tool: Tool, dispatch: Dispatch) -> BetaFunctionTool[Callable[..., str]]:
    """A tool the runner can call, sent exactly as `tool` is written and run through `dispatch`."""
    name = cast(str, tool["name"])

    def run(**arguments: object) -> str:
        content, is_error = dispatch(name, arguments)
        if is_error:
            raise ToolError(content)
        return content

    return beta_tool(
        run,
        name=name,
        description=cast("str | None", tool.get("description")),
        input_schema=cast(InputSchema, tool["input_schema"]),
        strict=cast("bool | None", tool.get("strict")),
    )


class Streaming:
    """An Anthropic stream, read as text and then as one `Reply`."""

    def __init__(self, stream: MessageStream) -> None:
        self._stream = stream

    @property
    def text_stream(self) -> Iterator[str]:
        return self._stream.text_stream

    @property
    def raw(self) -> MessageStream:
        """The SDK's own stream, for a caller that wants its events rather than the text."""
        return self._stream

    def final(self) -> Reply:
        return reply_of(self._stream.get_final_message())


class ToolSession:
    """A tool-using conversation on the SDK's runner: turns out, tool results back in.

    The wire half of `llm.run_tools` — iterating, running the tools, and sending a last answer with
    the same tools on the wire so the cache still holds. `llm` keeps admitting and billing.
    """

    def __init__(
        self,
        client: AnthropicClient,
        settings: Client,
        request: Request,
        dispatch: Dispatch,
        max_turns: int,
    ) -> None:
        self._settings = settings
        tools = request.tools or []
        runnable = [_runnable(tool, dispatch) for tool in tools if is_client_tool(tool)]
        server = [cast(BetaToolUnionParam, tool) for tool in tools if not is_client_tool(tool)]
        # What the runner puts on the wire: its own tools first, then the ones the API runs.
        self._wire: list[BetaToolUnionParam] = [*(tool.to_dict() for tool in runnable), *server]
        self._client = client
        self._runner = client.beta.messages.tool_runner(
            model=model_of(request, settings),
            max_tokens=request.max_tokens,
            messages=cast(list[BetaMessageParam], request.messages),
            system=cast("str | list[BetaTextBlockParam] | Omit", system_param(request)),
            thinking=cast("BetaThinkingConfigParam | Omit", or_omit(request.thinking)),
            output_config=cast("BetaOutputConfigParam | Omit", or_omit(request.output_config)),
            cache_control=cast(
                "BetaCacheControlEphemeralParam | Omit", or_omit(request.cache_control)
            ),
            tools=[*runnable, *server],
            max_iterations=max_turns,
        )

    def __iter__(self) -> Iterator[Reply]:
        for message in self._runner:
            yield reply_of(message)

    def tool_results(self) -> Msg | None:
        """Run the tools this turn asked for; the runner reuses these rather than rerun them."""
        results = self._runner.generate_tool_call_response()
        return None if results is None else cast(Msg, results)

    def answer(self, request: Request) -> Reply:
        """A last turn sent as the runner sends one, bar the messages and the cap."""
        return reply_of(
            self._client.beta.messages.parse(
                model=model_of(request, self._settings),
                max_tokens=request.max_tokens,
                messages=cast(list[BetaMessageParam], request.messages),
                system=cast("str | list[BetaTextBlockParam] | Omit", system_param(request)),
                thinking=cast("BetaThinkingConfigParam | Omit", or_omit(request.thinking)),
                output_config=cast("BetaOutputConfigParam | Omit", or_omit(request.output_config)),
                cache_control=cast(
                    "BetaCacheControlEphemeralParam | Omit", or_omit(request.cache_control)
                ),
                tools=self._wire,
            )
        )


class Anthropic:
    """The Anthropic wire, holding one SDK client.

    Built by `adapters.build` from a provider name, over a client handed in or one of its own.
    """

    name = "anthropic"
    supports = SUPPORTS

    def __init__(self, settings: Client, sdk: object | None = None) -> None:
        # `sdk` is typed loosely because `adapters.build` hands every adapter the same argument:
        # what a vendor client *is* differs per wire, and only this class knows which it wants.
        self.settings = settings
        self._client = cast("AnthropicClient | None", sdk)

    @property
    def client(self) -> AnthropicClient:
        """Built on first use, so importing this module needs no credentials."""
        if self._client is None:
            self._client = get_client(self.settings.base_url)
        return self._client

    def count(self, request: Request, schema: type[BaseModel] | None = None) -> int:
        """The input tokens this request would send, schema included. Free: nothing is generated."""
        return self.client.messages.count_tokens(
            model=model_of(request, self.settings),
            messages=cast(Sequence[MessageParam], request.messages),
            system=system_param(request),
            tools=cast("Sequence[ToolUnionParam] | Omit", or_omit(request.tools)),
            thinking=cast("ThinkingConfigParam | Omit", or_omit(request.thinking)),
            output_config=cast("OutputConfigParam | Omit", or_omit(request.output_config)),
            output_format=or_omit(schema),
            cache_control=cast("CacheControlEphemeralParam | Omit", or_omit(request.cache_control)),
        ).input_tokens

    def send(self, request: Request) -> Reply:
        """One request, one reply."""
        return reply_of(
            self.client.messages.create(
                model=model_of(request, self.settings),
                max_tokens=request.max_tokens,
                messages=cast(Sequence[MessageParam], request.messages),
                system=system_param(request),
                tools=cast("Sequence[ToolUnionParam] | Omit", or_omit(request.tools)),
                thinking=cast("ThinkingConfigParam | Omit", or_omit(request.thinking)),
                output_config=cast("OutputConfigParam | Omit", or_omit(request.output_config)),
                cache_control=cast(
                    "CacheControlEphemeralParam | Omit", or_omit(request.cache_control)
                ),
            )
        )

    def parse[Record: BaseModel](
        self, request: Request, schema: type[Record], cache_body: dict[str, object] | None = None
    ) -> Parsed[Record]:
        """One request for a `schema` record, which the SDK reads as it streams in.

        Raises `Unreadable` when the record does not validate, which the SDK detects while reading
        and so before any usage is seen; `llm` prices that call from its own count.
        """
        try:
            message = self.client.messages.parse(
                model=model_of(request, self.settings),
                max_tokens=request.max_tokens,
                messages=cast(Sequence[MessageParam], request.messages),
                system=system_param(request),
                tools=cast("Sequence[ToolUnionParam] | Omit", or_omit(request.tools)),
                thinking=cast("ThinkingConfigParam | Omit", or_omit(request.thinking)),
                output_config=cast("OutputConfigParam | Omit", or_omit(request.output_config)),
                output_format=schema,
                # The SDK's parse has no cache_control parameter; the API takes it all the same.
                extra_body=cache_body,
            )
        except ValidationError as exc:
            raise Unreadable(f"{request.step}: the record did not validate") from exc
        return as_parsed(message)

    @contextmanager
    def streamed(self, request: Request) -> Iterator[Streaming]:
        """One request as a stream; the caller reads it and `llm` bills its final reply."""
        with self.client.messages.stream(
            model=model_of(request, self.settings),
            max_tokens=request.max_tokens,
            messages=cast(Sequence[MessageParam], request.messages),
            system=system_param(request),
            tools=cast("Sequence[ToolUnionParam] | Omit", or_omit(request.tools)),
            thinking=cast("ThinkingConfigParam | Omit", or_omit(request.thinking)),
            output_config=cast("OutputConfigParam | Omit", or_omit(request.output_config)),
            cache_control=cast("CacheControlEphemeralParam | Omit", or_omit(request.cache_control)),
        ) as stream:
            yield Streaming(stream)

    def tools(self, request: Request, dispatch: Dispatch, max_turns: int) -> ToolSession:
        """A tool-using conversation, which `llm` drives one admitted turn at a time."""
        return ToolSession(self.client, self.settings, request, dispatch, max_turns)


def has_credentials() -> bool:
    """True if the SDK will find something to authenticate with."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def get_client(base_url: str | None = None) -> AnthropicClient:
    """Build the SDK client. Credentials are this wire's business, not `config`'s.

    Set ANTHROPIC_WORKSPACE_ID only for an org-level key; the API refuses one without the header.
    """
    if not has_credentials():
        raise SystemExit(
            "No Anthropic credentials found.\n"
            "  Option 1: copy .env.example to .env and set ANTHROPIC_API_KEY\n"
            "  Option 2: install the `ant` CLI and run `ant auth login`\n"
        )
    headers: dict[str, str] = {}
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        headers["anthropic-workspace-id"] = workspace_id
    if base_url is not None:
        return anthropic.Anthropic(default_headers=headers, base_url=base_url)
    return anthropic.Anthropic(default_headers=headers)
