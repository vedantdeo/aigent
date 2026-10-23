"""The MCP server through the SDK's own client, in process: what it lists, what each tool returns
and refuses, and that the real searches wait for the first call."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import anyio
import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent, Tool

from aigent import mcp_server
from aigent.analyst import filings
from aigent.analyst.filings import SEARCH_FILINGS_TOOL
from aigent.analyst.fundamentals import FUNDAMENTALS_TOOL
from aigent.config import MCP_SERVER_LOG_LEVEL, MCP_SERVER_NAME
from aigent.facts.lookup import Lookup
from aigent.mcp_server import LazyFilings, LazySearch, build_server
from aigent.report_tools import SEARCH_TOOL
from aigent.retrieval.chunk import Chunk
from aigent.tools import CALCULATOR_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, fence
from aigent.workflows.reports import REPORTS, Search

from .conftest import PASSAGES, FakeFilings, FakeSearch


def _with_client[Result](
    search: Search,
    use: Callable[[Client], Awaitable[Result]],
    sandbox: Path | None = None,
    filing_search: filings.Search | None = None,
    lookup: Lookup | None = None,
) -> Result:
    async def run() -> Result:
        asked = filing_search or FakeFilings()
        box = {} if sandbox is None else {"sandbox": sandbox}
        server = build_server(search, asked, lookup=lookup, **box)
        async with Client(server) as client:
            return await use(client)

    return anyio.run(run)


def _call(
    search: Search,
    arguments: dict[str, object],
    tool: str = "search_reports",
    sandbox: Path | None = None,
    filing_search: filings.Search | None = None,
    lookup: Lookup | None = None,
) -> CallToolResult:
    return _with_client(
        search, lambda client: client.call_tool(tool, arguments), sandbox, filing_search, lookup
    )


def _text(result: CallToolResult) -> str:
    return "".join(cast(TextContent, block).text for block in result.content)


def test_the_server_offers_six_tools_described_as_the_agent_sees_them(
    search: FakeSearch,
) -> None:
    async def listed(client: Client) -> tuple[list[Tool], str | None, str]:
        tools = (await client.list_tools()).tools
        named = client.server_info.name if client.server_info else None
        return tools, named, client.instructions or ""

    listing, name, instructions = _with_client(search, listed)
    tools = {tool.name: tool for tool in listing}

    assert name == MCP_SERVER_NAME
    assert build_server(search, FakeFilings()).settings.log_level == MCP_SERVER_LOG_LEVEL
    assert all(company in instructions for company in REPORTS.values()), instructions
    assert {named: tool.description for named, tool in tools.items()} == {
        "search_reports": SEARCH_TOOL.get("description"),
        "search_filings": SEARCH_FILINGS_TOOL.get("description"),
        "fundamentals": FUNDAMENTALS_TOOL.get("description"),
        "calculate": CALCULATOR_TOOL.get("description"),
        "read_file": READ_FILE_TOOL.get("description"),
        "write_file": WRITE_FILE_TOOL.get("description"),
    }
    assert tools["read_file"].annotations and tools["read_file"].annotations.read_only_hint
    assert tools["write_file"].annotations and not tools["write_file"].annotations.read_only_hint
    assert tools["write_file"].input_schema["required"] == ["file_path", "content"]
    assert tools["search_filings"].input_schema["required"] == ["query"], "scope defaults to all"
    assert tools["fundamentals"].input_schema["required"] == ["company", "year", "what"]
    for served, ours in (
        ("search_filings", SEARCH_FILINGS_TOOL),
        ("fundamentals", FUNDAMENTALS_TOOL),
    ):
        shown = cast(dict[str, dict[str, object]], dict(ours["input_schema"])["properties"])
        served_as = cast(dict[str, dict[str, object]], tools[served].input_schema["properties"])
        allowed = {field: spec["enum"] for field, spec in shown.items() if "enum" in spec}
        assert {field: served_as[field]["enum"] for field in allowed} == allowed, served
    tool = tools["search_reports"]
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


@pytest.mark.parametrize(
    ("arguments", "asked", "doc_ids"),
    [
        pytest.param({"query": "dividend"}, ("dividend", "all", "all"), set(PASSAGES), id="all"),
        pytest.param(
            {"query": " dividend\n", "company": "RELIANCE", "year": "2025"},
            ("dividend", "RELIANCE", "2025"),
            {"RELIANCE-FY25"},
            id="one company and year, the query trimmed",
        ),
        pytest.param(
            {"query": "dividend", "company": "ITC", "year": "2024"},
            ("dividend", "ITC", "2024"),
            set[str](),
            id="a report with no passage",
        ),
    ],
)
def test_a_filings_search_returns_fenced_passages_from_the_scope_asked(
    arguments: dict[str, object],
    asked: tuple[str, str, str],
    doc_ids: set[str],
    search: FakeSearch,
    filing_search: FakeFilings,
) -> None:
    result = _call(search, arguments, "search_filings", filing_search=filing_search)

    text = _text(result)
    assert not result.is_error and filing_search.asked == [asked], text
    assert text.startswith("<untrusted") and ("No passages matched" in text) is (not doc_ids)
    cited = {chunk.doc_id for chunks in PASSAGES.values() for chunk in chunks if chunk.id in text}
    assert cited == doc_ids, text


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param({"query": "  "}, "empty", id="an empty query"),
        pytest.param({"query": "q", "company": "INFY"}, "INFY", id="a company not held"),
        pytest.param({"query": "q", "year": "2019"}, "2019", id="a year not held"),
        pytest.param({"query": "q", "year": 2025}, "2025", id="a year sent as a number"),
    ],
)
def test_a_bad_filings_search_is_refused_and_nothing_is_searched(
    arguments: dict[str, object], says: str, search: FakeSearch, filing_search: FakeFilings
) -> None:
    result = _call(search, arguments, "search_filings", filing_search=filing_search)

    assert result.is_error and says in _text(result), _text(result)
    assert filing_search.asked == []


def test_a_filings_search_that_fails_is_an_error_not_an_answer(search: FakeSearch) -> None:
    def broken(query: str, /, *, company: str = "all", year: str = "all") -> list[Chunk]:
        raise ValueError("the index is missing")

    result = _call(search, {"query": "q"}, "search_filings", filing_search=broken)

    assert result.is_error and "the index is missing" in _text(result), _text(result)
    assert ": Error:" not in _text(result), "the SDK prefixes its own Error"


def test_a_fundamentals_lookup_returns_the_facts_and_what_to_cite(
    search: FakeSearch, facts_lookup: Lookup
) -> None:
    arguments: dict[str, object] = {
        "company": "ITC",
        "year": "2025",
        "what": "  revenue from operations  ",
        "period": "Q4",
    }

    result = _call(search, arguments, "fundamentals", lookup=facts_lookup)

    found = cast(dict[str, list[dict[str, object]]], json.loads(_text(result)))["facts"][0]
    assert not result.is_error, _text(result)
    assert found["cite_as"] == "FACT:ITC:2025:consolidated:Q4:RevenueFromOperations", found
    assert found["value"] == 204.0, found


@pytest.mark.parametrize(
    ("tool", "arguments", "says"),
    [
        pytest.param(
            "fundamentals",
            {"company": "INFY", "year": "2025", "what": "revenue"},
            "INFY",
            id="a company the database lacks",
        ),
        pytest.param(
            "fundamentals",
            {"company": "ITC", "year": "2019", "what": "revenue"},
            "2019",
            id="a year the database lacks",
        ),
        pytest.param(
            "fundamentals",
            {"company": "ITC", "year": "2025", "what": "revenue", "period": "Q5"},
            "Q5",
            id="a period that is not one",
        ),
        pytest.param(
            "fundamentals",
            {"company": "ITC", "year": "2025", "what": ""},
            "empty",
            id="no figure named, refused by the input guard",
        ),
        pytest.param(
            "fundamentals",
            {"company": "WIPRO", "year": "2025", "what": "revenue from operations"},
            "Closest labels filed",
            id="a figure the database does not hold",
        ),
        pytest.param("calculate", {"expression": "__import__('os')"}, "", id="code, not sums"),
    ],
)
def test_a_bad_lookup_or_sum_is_refused_with_its_reason(
    search: FakeSearch, facts_lookup: Lookup, tool: str, arguments: dict[str, object], says: str
) -> None:
    result = _call(search, arguments, tool, lookup=facts_lookup)

    assert result.is_error and says in _text(result), _text(result)
    assert ": Error:" not in _text(result), "the SDK prefixes its own Error"


def test_calculate_returns_the_result(search: FakeSearch) -> None:
    result = _call(search, {"expression": "(14.35 - 13.75) / 13.75 * 100"}, "calculate")

    assert not result.is_error and float(_text(result)) == pytest.approx(4.3636, abs=1e-3)


def test_the_filings_search_refuses_a_scope_before_building_then_builds_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[bool] = []

    def build_filings_search(cache: bool) -> filings.Search:
        built.append(cache)
        return FakeFilings()

    monkeypatch.setattr(mcp_server, "build_filings_search", build_filings_search)
    lazy = LazyFilings()

    with pytest.raises(ValueError, match="no company"):
        lazy("q", company="INFY")
    assert built == [], "neither starting nor a refused scope loads a model"

    lazy("q")
    lazy("q", company="ITC", year="2025")

    assert built == [True], "built once, from the cached corpus and vectors"


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


def test_a_read_arrives_fenced_from_the_sandbox_it_was_given(search: FakeSearch, box: Path) -> None:
    result = _call(search, {"file_path": "notes.txt"}, "read_file", box)

    assert not result.is_error, _text(result)
    assert _text(result) == fence("read_file", "hello from the sandbox\n")


def test_a_write_creates_a_file_the_next_read_returns(search: FakeSearch, box: Path) -> None:
    async def write_then_read(client: Client) -> tuple[CallToolResult, CallToolResult]:
        wrote = await client.call_tool("write_file", {"file_path": "out/memo.md", "content": "hi"})
        return wrote, await client.call_tool("read_file", {"file_path": "out/memo.md"})

    wrote, read = _with_client(search, write_then_read, box)

    assert not wrote.is_error and _text(wrote) == "Wrote 2 characters to out/memo.md", _text(wrote)
    assert (box / "out" / "memo.md").read_text(encoding="utf-8") == "hi"
    assert _text(read) == fence("read_file", "hi"), _text(read)


@pytest.mark.parametrize(
    ("tool", "arguments", "says"),
    [
        pytest.param(
            "read_file", {"file_path": "missing.txt"}, "File not found", id="no such file"
        ),
        pytest.param("read_file", {"file_path": "../outside.txt"}, "escapes", id="a read out"),
        pytest.param(
            "write_file",
            {"file_path": "../planted.txt", "content": "x"},
            "escapes",
            id="a write out",
        ),
        pytest.param(
            "write_file",
            {"file_path": "notes.txt", "content": "x"},
            "already exists",
            id="a write over an existing file",
        ),
    ],
)
def test_a_bad_file_call_is_refused_with_its_reason_and_changes_nothing(
    search: FakeSearch, box: Path, tool: str, arguments: dict[str, object], says: str
) -> None:
    result = _call(search, arguments, tool, box)

    assert result.is_error and says in _text(result), _text(result)
    assert ": Error:" not in _text(result), "the SDK prefixes its own Error"
    assert (box / "notes.txt").read_text(encoding="utf-8") == "hello from the sandbox\n"
    assert not (box.parent / "planted.txt").exists()
