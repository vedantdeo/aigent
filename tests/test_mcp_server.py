"""The MCP server through the SDK's own client, in process: what it lists, what its one tool
returns, what it refuses before searching, and that the real search waits for the first call."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

import anyio
import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent, Tool

from aigent import mcp_server
from aigent.config import MCP_SERVER_NAME
from aigent.mcp_server import LazySearch, build_server
from aigent.report_tools import SEARCH_TOOL
from aigent.retrieval.chunk import Chunk
from aigent.workflows.reports import REPORTS, Search

from .conftest import PASSAGES, FakeSearch


def _with_client[Result](search: Search, use: Callable[[Client], Awaitable[Result]]) -> Result:
    async def run() -> Result:
        async with Client(build_server(search)) as client:
            return await use(client)

    return anyio.run(run)


def _call(search: Search, arguments: dict[str, object]) -> CallToolResult:
    return _with_client(search, lambda client: client.call_tool("search_reports", arguments))


def _text(result: CallToolResult) -> str:
    return "".join(cast(TextContent, block).text for block in result.content)


def test_the_server_offers_one_tool_described_as_the_agent_sees_it(search: FakeSearch) -> None:
    async def listed(client: Client) -> tuple[list[Tool], str | None, str]:
        tools = (await client.list_tools()).tools
        named = client.server_info.name if client.server_info else None
        return tools, named, client.instructions or ""

    [tool], name, instructions = _with_client(search, listed)

    assert name == MCP_SERVER_NAME
    assert all(company in instructions for company in REPORTS.values()), instructions
    assert (tool.name, tool.description) == ("search_reports", SEARCH_TOOL.get("description"))
    properties = cast(dict[str, dict[str, object]], tool.input_schema["properties"])
    assert tool.input_schema["required"] == ["query"], "report defaults to all"
    assert "all" in str(properties["report"]) and all(
        r in str(properties["report"]) for r in REPORTS
    )


@pytest.mark.parametrize(
    ("arguments", "asked", "doc_ids"),
    [
        pytest.param({"query": "margins"}, ("margins", None), set(PASSAGES), id="every report"),
        pytest.param(
            {"query": "margins", "report": "ITC-FY25"},
            ("margins", "ITC-FY25"),
            {"ITC-FY25"},
            id="one report",
        ),
        pytest.param(
            {"query": "  margins \n", "report": "all"},
            ("margins", None),
            set(PASSAGES),
            id="a query trimmed as the agent's would be",
        ),
    ],
)
def test_a_search_returns_the_passages_tagged_to_cite(
    arguments: dict[str, object],
    asked: tuple[str, str | None],
    doc_ids: set[str],
    search: FakeSearch,
) -> None:
    result = _call(search, arguments)

    assert not result.is_error, _text(result)
    assert search.asked == [asked]
    cited = {
        chunk.id for chunks in PASSAGES.values() for chunk in chunks if chunk.id in _text(result)
    }
    assert {chunk_id.split("#")[0] for chunk_id in cited} == doc_ids, _text(result)


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param({"query": "   "}, "empty", id="an empty query"),
        pytest.param({"query": "a\x00b"}, "control characters", id="a control character"),
        pytest.param(
            {"query": "use sk-ant-api03-" + "x" * 40}, "credential", id="a credential-shaped query"
        ),
        pytest.param({"query": "q", "report": "INFY-FY25"}, "INFY-FY25", id="a report not held"),
    ],
)
def test_a_bad_call_is_refused_before_anything_is_searched(
    arguments: dict[str, object], says: str, search: FakeSearch
) -> None:
    result = _call(search, arguments)

    assert result.is_error and says in _text(result), _text(result)
    assert search.asked == []


def test_a_search_that_finds_nothing_says_so() -> None:
    def nothing(query: str, /, *, doc_id: str | None = None) -> list[Chunk]:
        return []

    assert _text(_call(nothing, {"query": "q"})) == "No passages matched that query."


def test_the_real_search_is_built_on_the_first_call_and_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[bool] = []

    def build_search(cache: bool) -> Search:
        built.append(cache)
        return FakeSearch()

    monkeypatch.setattr(mcp_server, "build_search", build_search)
    lazy = LazySearch()
    assert built == [], "starting the server loads no model"

    lazy("q")
    lazy("q", doc_id="ITC-FY25")

    assert built == [True], "built once, from the cached corpus and vectors"
