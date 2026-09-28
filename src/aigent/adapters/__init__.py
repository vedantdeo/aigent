"""The clients we can talk to, and the wire each one speaks. The only place an SDK is imported.

A **client** is a named endpoint — `anthropic` today, a local `mlx_lm.server` next. An **adapter**
is the class that speaks its wire: it converts a neutral request into that wire's parameters and
its reply back into a `messages.Reply`, and declares in `supports` what the wire can do. Many
clients can share one adapter — anything OpenAI-compatible speaks the same wire — but a client
maps to exactly one, so naming the client is enough to pick it.

`llm` holds the adapter for its client and calls it for every request; everything else a call needs
— counting, admission, billing, the trace, concurrency — stays in `llm`. A new client is a row in
`CLIENTS`, not a branch through `llm`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from aigent.adapters.anthropic import Anthropic
from aigent.adapters.cfg_anthropic import CLIENT as ANTHROPIC
from aigent.adapters.cfg_local import CLIENTS as LOCAL_CLIENTS
from aigent.adapters.client import Client
from aigent.adapters.openai import OpenAI
from aigent.messages import Msg, Parsed, Reply

if TYPE_CHECKING:
    from aigent.llm import Dispatch, Request


class Streamed(Protocol):
    """A reply arriving in pieces: text as it comes, then the whole of it once."""

    @property
    def text_stream(self) -> Iterator[str]: ...

    def final(self) -> Reply: ...


class ToolSession(Protocol):
    """A tool-using conversation: turns out, tool results back in, and a last answer on demand."""

    def __iter__(self) -> Iterator[Reply]: ...

    def tool_results(self) -> Msg | None: ...

    def answer(self, request: Request) -> Reply: ...


class Adapter(Protocol):
    """What `llm` needs of a wire. Everything here sends; nothing here counts money or bills."""

    name: str
    supports: frozenset[str]

    @property
    def client(self) -> object:
        """The provider's own client, so one connection can be shared across several `Llm`s."""
        ...

    def count(self, request: Request, schema: type[BaseModel] | None = None) -> int: ...

    def send(self, request: Request) -> Reply: ...

    def parse[Record: BaseModel](
        self, request: Request, schema: type[Record], cache_body: dict[str, object] | None = None
    ) -> Parsed[Record]: ...

    def streamed(self, request: Request) -> AbstractContextManager[Streamed]: ...

    def tools(self, request: Request, dispatch: Dispatch, max_turns: int) -> ToolSession: ...


# Wire name → the class that speaks it, in one canonical order.
WIRES: dict[str, Callable[[Client, object | None], Adapter]] = {
    "anthropic": Anthropic,
    "openai": OpenAI,
}

# Client name → its settings, from the `cfg_<name>` module beside this one. Several clients may
# share a wire; each names exactly one, so the client's name is enough to pick the adapter.
CLIENTS: dict[str, Client] = {"anthropic": ANTHROPIC, **LOCAL_CLIENTS}


def spec(client: str) -> Client:
    """The named client's settings. Raises rather than guessing which endpoint was meant."""
    try:
        return CLIENTS[client]
    except KeyError:
        known = ", ".join(sorted(CLIENTS))
        raise ValueError(f"no client named {client!r}; there is {known}") from None


def build(client: str, sdk: object | None = None) -> Adapter:
    """The adapter for the named client, over `sdk` if a vendor client is handed in.

    Raises on an unknown client rather than guessing a default, since guessing would send a request
    somewhere nobody asked for.
    """
    settings = spec(client)
    return WIRES[settings.wire](settings, sdk)
