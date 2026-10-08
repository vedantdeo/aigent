"""The annual reports' search and the sandbox's files as an MCP server, over stdio.

Three tools for other MCP clients, such as Claude Code: `search_reports`, answering through
`report_tools.ReportSearch` with its query through the agent's input guard, and `read_file` and
`write_file`, through `tools.execute_tool`, so a read arrives fenced and a write cannot overwrite.

uv run python -m aigent.mcp_server
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from aigent import guardrails
from aigent.config import MCP_SERVER_NAME
from aigent.errors import GuardrailTripped
from aigent.report_tools import SEARCH_TOOL, ReportSearch
from aigent.retrieval.chunk import Chunk
from aigent.tools import READ_FILE_TOOL, WRITE_FILE_TOOL, execute_tool
from aigent.tools_config import SANDBOX
from aigent.workflows.reports import CATALOGUE, DocId, Search, build_search

Report = DocId | Literal["all"]


def build_server(search: Search, sandbox: Path = SANDBOX) -> MCPServer:
    """A server that searches the reports through `search` and reads and writes inside `sandbox`."""
    server = MCPServer(
        MCP_SERVER_NAME,
        instructions=(
            f"Search three Indian companies' FY25 annual reports:\n{CATALOGUE}\n"
            "Read files from, and create new files in, aigent's sandbox directory."
        ),
    )
    run = ReportSearch(search)

    @server.tool(name="search_reports", description=SEARCH_TOOL.get("description", ""))
    def search_reports(query: str, report: Report = "all") -> str:
        try:
            query = guardrails.check_task(query)
        except GuardrailTripped as refused:
            raise ToolError(str(refused)) from refused  # a ToolError's text reaches the client
        return run({"query": query, "report": report})

    def dispatch(name: str, tool_input: dict[str, object]) -> str:
        content, is_error = execute_tool(name, tool_input, sandbox=sandbox, writes=True)
        if is_error:
            raise ToolError(content.removeprefix("Error: "))  # the SDK adds its own
        return content

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
