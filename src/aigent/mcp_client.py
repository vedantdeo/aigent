"""An MCP server's tools in the agent's loop: listed as the schemas a model is shown, and called
through an `llm.Dispatch`, so a tool we did not write is counted, fenced and traced like our own.

    with connect(OWN_SERVER, fenced=False) as served:
        llm.run_tools(replace(request, tools=served.tools), served.dispatch)
"""

from __future__ import annotations

import sys
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from anthropic.types import ToolParam
from anyio.from_thread import start_blocking_portal
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent

from aigent.config import MCP_CALL_TIMEOUT_SECONDS, MCP_MAX_RESULT_CHARS
from aigent.llm.core import Dispatch
from aigent.tools import fence

Server = MCPServer | StdioServerParameters | str  # in process, a command to launch, or a URL
# aigent's own server over stdio; it fences what it returns, so connect with `fenced=False`.
OWN_SERVER = StdioServerParameters(command=sys.executable, args=["-m", "aigent.mcp_server"])


@dataclass(frozen=True)
class Served:
    """A connected server's tools: what a model is shown, and the dispatcher that calls them."""

    tools: list[ToolParam]
    dispatch: Dispatch


@contextmanager
def connect(
    server: Server,
    *,
    only: Collection[str] | None = None,
    fenced: bool = True,
    timeout: float = MCP_CALL_TIMEOUT_SECONDS,
    max_chars: int = MCP_MAX_RESULT_CHARS,
) -> Iterator[Served]:
    """The tools of `server`, or just those named in `only`, for as long as the block runs.

    A call never raises: a refusal, a timeout or a dead server comes back as an error result. Text
    is cut at `max_chars` and, unless the server is ours and fences its own (`fenced=False`),
    fenced as untrusted. ValueError if `only` names a tool the server does not offer.
    """
    raised: BaseException | None = None
    # the SDK is async and the agent's loop is not, so the session runs on a portal's thread
    with (
        start_blocking_portal() as portal,
        portal.wrap_async_context_manager(Client(server, read_timeout_seconds=timeout)) as client,
    ):
        offered = {tool.name: tool for tool in portal.call(client.list_tools).tools}
        chosen = [tool for name, tool in offered.items() if only is None or name in only]
        missing = sorted(set(only or ()) - set(offered))

        def dispatch(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
            if not any(tool.name == name for tool in chosen):
                return f"Error: {name} is not available here", True
            try:
                result = portal.call(client.call_tool, name, arguments)
            except Exception as failed:  # this call's failure, not the run's
                return f"Error: {name} failed: {type(failed).__name__}: {failed}", True
            text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
            if len(text) > max_chars:
                text = f"{text[:max_chars]}\n[truncated at {max_chars} characters]"
            return (fence(name, text) if fenced else text), bool(result.is_error)

        shown: list[ToolParam] = [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in chosen
        ]
        if missing:
            has = ", ".join(sorted(offered))
            raised = ValueError(f"the server offers no {', '.join(missing)}; it has {has}")
        else:
            try:
                yield Served(shown, dispatch)
            except BaseException as from_the_block:  # the session's task group would wrap it
                raised = from_the_block
    if raised is not None:
        raise raised
