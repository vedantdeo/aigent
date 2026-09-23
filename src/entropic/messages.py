"""What a request carries and a reply returns, in this repo's types rather than a provider's.

An adapter converts these to a provider's wire format and back, so nothing above `llm` imports a
provider SDK. The shapes deliberately mirror Anthropic's, which is the richer of the two: content
blocks, a separate system prompt, and cache markers all survive a round trip, while the OpenAI
adapter flattens what it can and refuses what it cannot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel

Role = Literal["user", "assistant"]

# One piece of a message: text, a tool call, a tool result. Keys vary by type, so it stays a
# mapping rather than a union of every block a provider defines. JSON-shaped, but typed loosely
# enough that a provider's own parameter dicts satisfy it without conversion.
Block = Mapping[str, object]

# A tool as the model is told about it: name, description, input schema.
Tool = Mapping[str, object]


class Msg(TypedDict):
    """One turn of a conversation."""

    role: Role
    content: str | Sequence[Block]


class Cache(TypedDict):
    """A marker asking the provider to cache everything up to this point."""

    type: Literal["ephemeral"]
    ttl: NotRequired[str]


class Thinking(TypedDict):
    """Whether the model may think before answering, and how much."""

    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: NotRequired[int]
    display: NotRequired[str]


class OutputConfig(TypedDict):
    """How hard the model should work on this one call."""

    effort: Literal["low", "medium", "high", "xhigh"]


def is_client_tool(tool: Tool) -> bool:
    """A tool we run ourselves, as opposed to one the provider runs or defines."""
    return tool.get("type") in (None, "custom")


@dataclass(frozen=True)
class Usage:
    """What one call used, in tokens, plus any server-side searches it ran."""

    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    web_searches: int = 0


@dataclass(frozen=True)
class Reply:
    """One model reply, whichever provider sent it.

    `blocks` keeps what `text` throws away — tool calls, search results — for the callers that
    read them; `raw` is the provider's own object, for the Anthropic-only paths that still need it.
    """

    text: str
    usage: Usage
    stop_reason: str | None
    model: str
    blocks: tuple[Block, ...] = ()
    raw: object = None


@dataclass(frozen=True)
class Parsed[Record: BaseModel](Reply):
    """A reply that was asked for a record. `parsed` is None when none could be read."""

    parsed: Record | None = None
