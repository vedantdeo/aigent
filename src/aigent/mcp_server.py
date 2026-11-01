"""The filings' search, the facts database, the calculator and the sandbox's files as an MCP
server, over stdio.

Six tools for an MCP client, the capstone's generalist among them: `search_reports` (the FY25
reports) and `search_filings` (both years, scoped by company and year), each query through the
agent's input guard; `fundamentals`; and `calculate`, `read_file` and `write_file` through
`tools.execute_tool`, so a read arrives fenced and a write cannot overwrite.

uv run python -m aigent.mcp_server
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from aigent import guardrails
from aigent.analyst import filings
from aigent.analyst.filings import SEARCH_FILINGS_TOOL, FilingSearch, build_filings_search
from aigent.analyst.fundamentals import (
    COMPARE_TOOL,
    FUNDAMENTALS_TOOL,
    Company,
    FinancialYear,
    FundamentalsTool,
)
from aigent.config import MCP_SERVER_LOG_LEVEL, MCP_SERVER_NAME
from aigent.errors import GuardrailTripped
from aigent.facts.lookup import Basis, Lookup, Period, concept_index
from aigent.facts.store import FACTS_DB
from aigent.report_tools import SEARCH_TOOL, ReportSearch
from aigent.retrieval.chunk import Chunk
from aigent.tools import CALCULATOR_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, execute_tool
from aigent.tools_config import SANDBOX
from aigent.workflows.reports import CATALOGUE, DocId, Search, build_search

Report = DocId | Literal["all"]


def _guarded(query: str) -> str:
    try:
        return guardrails.check_task(query)
    except GuardrailTripped as refused:
        raise ToolError(str(refused)) from refused  # a ToolError's text reaches the client


def _answered(content: str) -> str:
    if content.startswith("Error: "):
        raise ToolError(content.removeprefix("Error: "))  # the SDK adds its own
    return content


def build_server(
    search: Search,
    filing_search: filings.Search,
    sandbox: Path = SANDBOX,
    lookup: Lookup | None = None,
) -> MCPServer:
    """A server that searches the FY25 reports through `search` and all six filings through
    `filing_search`, looks figures up through `lookup` (the facts database by default), calculates,
    and reads and writes inside `sandbox`."""
    server = MCPServer(
        MCP_SERVER_NAME,
        instructions=(
            f"Search three Indian companies' FY25 annual reports:\n{CATALOGUE}\n"
            "Search their FY24 reports too, by company and year, look up nine companies' "
            "reported figures, and calculate. Read files from, and create new files in, aigent's "
            "sandbox directory."
        ),
        log_level=MCP_SERVER_LOG_LEVEL,
    )
    run = ReportSearch(search)
    run_filings = FilingSearch(filing_search)
    run_fundamentals = FundamentalsTool(_session_lookup() if lookup is None else lookup)

    @server.tool(name="search_reports", description=SEARCH_TOOL.get("description", ""))
    def search_reports(query: str, report: Report = "all") -> str:
        return run({"query": _guarded(query), "report": report})

    @server.tool(name="search_filings", description=SEARCH_FILINGS_TOOL.get("description", ""))
    def search_filings(
        query: str, company: filings.Company = "all", year: filings.Year = "all"
    ) -> str:
        return _answered(run_filings({"query": _guarded(query), "company": company, "year": year}))

    @server.tool(
        name="fundamentals",
        description=FUNDAMENTALS_TOOL.get("description", ""),
        annotations=ToolAnnotations(read_only_hint=True),
    )
    def fundamentals(
        company: Company,
        year: FinancialYear,
        what: str,
        period: Period = "year",
        basis: Basis = "consolidated",
        segment: str | None = None,
    ) -> str:
        arguments: dict[str, object] = {
            "company": company,
            "year": year,
            "what": _guarded(what),
            "period": period,
            "basis": basis,
            "segment": segment,
        }
        return _answered(run_fundamentals(arguments))

    @server.tool(
        name="compare_unverified",
        description=COMPARE_TOOL.get("description", ""),
        annotations=ToolAnnotations(read_only_hint=True),
    )
    def compare_unverified(lookup_id: str, my_value: float, how: str) -> str:
        arguments: dict[str, object] = {
            "lookup_id": lookup_id,
            "my_value": my_value,
            "how": _guarded(how),
        }
        return _answered(run_fundamentals.compare(arguments))

    def dispatch(name: str, tool_input: dict[str, object]) -> str:
        content, is_error = execute_tool(name, tool_input, sandbox=sandbox, writes=True)
        if is_error:
            raise ToolError(content.removeprefix("Error: "))  # the SDK adds its own
        return content

    @server.tool(
        name="calculate",
        description=CALCULATOR_TOOL.get("description", ""),
        annotations=ToolAnnotations(read_only_hint=True),
    )
    def calculate(expression: str) -> str:
        return dispatch("calculate", {"expression": expression})

    @server.tool(
        name="read_file",
        description=READ_FILE_TOOL.get("description", ""),
        annotations=ToolAnnotations(read_only_hint=True),
    )
    def read_file(file_path: str) -> str:
        return dispatch("read_file", {"file_path": file_path})

    @server.tool(
        name="write_file",
        description=WRITE_FILE_TOOL.get("description", ""),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False),
    )
    def write_file(file_path: str, content: str) -> str:
        return dispatch("write_file", {"file_path": file_path, "content": content})

    return server


def _session_lookup() -> Lookup:
    """The facts database, its lookups logged under this server process."""
    now = datetime.now().astimezone()
    return Lookup(
        FACTS_DB,
        concept_index(),
        run_id=f"mcp-{now:%Y%m%dT%H%M%S}-{os.getpid()}",
        asked_by=MCP_SERVER_NAME,
    )


class LazySearch:
    """The real search, built on first use, so the server starts before the models load."""

    def __init__(self) -> None:
        self._search: Search | None = None
        self._building = threading.Lock()

    def __call__(self, query: str, /, *, doc_id: str | None = None) -> list[Chunk]:
        with self._building:
            if self._search is None:
                self._search = build_search(cache=True)
        return self._search(query, doc_id=doc_id)


class LazyFilings:
    """The six filings' search, built on first use; a scope it does not hold is refused first."""

    def __init__(self) -> None:
        self._search: filings.Search | None = None
        self._building = threading.Lock()

    def __call__(
        self, query: str, /, *, company: str = filings.ALL, year: str = filings.ALL
    ) -> list[Chunk]:
        filings.scope(company, year)
        with self._building:
            if self._search is None:
                self._search = build_filings_search(cache=True)
        return self._search(query, company=company, year=year)


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # read when a model first loads
    build_server(LazySearch(), LazyFilings()).run()


if __name__ == "__main__":
    main()
