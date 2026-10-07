"""The annual reports' search as an MCP server: one tool, `search_reports`, over stdio.

aigent's agent calls the same search in process; this serves it to other MCP clients, such as
Claude Code. The tool answers in the agent's words, through `report_tools.ReportSearch`, and takes
its query through the agent's input guard.

uv run python -m aigent.mcp_server
"""

from __future__ import annotations

import threading
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from aigent import guardrails
from aigent.config import MCP_SERVER_NAME
from aigent.errors import GuardrailTripped
from aigent.report_tools import SEARCH_TOOL, ReportSearch
from aigent.retrieval.chunk import Chunk
from aigent.workflows.reports import CATALOGUE, DocId, Search, build_search

Report = DocId | Literal["all"]


def build_server(search: Search) -> MCPServer:
    """A server whose one tool searches the reports through `search`."""
    server = MCPServer(
        MCP_SERVER_NAME,
        instructions=f"Search three Indian companies' FY25 annual reports:\n{CATALOGUE}",
    )
    run = ReportSearch(search)

    @server.tool(name="search_reports", description=SEARCH_TOOL.get("description", ""))
    def search_reports(query: str, report: Report = "all") -> str:
        try:
            query = guardrails.check_task(query)
        except GuardrailTripped as refused:
            raise ToolError(str(refused)) from refused  # a ToolError's text reaches the client
        return run({"query": query, "report": report})

    return server


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


def main() -> None:
    build_server(LazySearch()).run()


if __name__ == "__main__":
    main()
