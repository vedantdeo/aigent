"""The generalist's tools in one place: a server's tools over MCP, and the two specialists as tools
whose calls go through the router. Passages a search returns are kept as evidence, so a later call
can name them by id.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

from anthropic.types import ToolParam
from pydantic import JsonValue

from aigent.analyst.router import Router
from aigent.analyst.specialists import FigureJob, Invalid, Specialist
from aigent.extraction.figures import Figures
from aigent.llm.core import Dispatch

SEARCH, LOOKUP = "search_filings", "fundamentals"
_PASSAGE = re.compile(r'<passage id="([^"]+)" source="([^"]*)">\n(.*?)\n</passage>', re.S)

READ_HEADLINE_TOOL: ToolParam = {
    "name": "read_headline",
    "description": (
        "Read one Indian market results headline into a record: the company as an NSE ticker, the "
        "metric, the quarter, the direction and the percentage change it states. One headline at "
        "a time; a company the directory does not list is an error."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"headline": {"type": "string", "description": "The headline, one line."}},
        "required": ["headline"],
        "additionalProperties": False,
    },
    "strict": True,
}

EXTRACT_FIGURES_TOOL: ToolParam = {
    "name": "extract_figures",
    "description": (
        "Extract the figures that passages from search_filings state: each a metric, a value in "
        "full with any named scale applied, a unit and the financial year, tagged with the "
        "passage it came from. Reads each passage on its own and does no arithmetic across them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids of passages search_filings returned in this conversation.",
            },
            "wanted": {"type": "string", "description": "What to extract, in a few words."},
        },
        "required": ["chunk_ids", "wanted"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass(frozen=True)
class Passage:
    """One passage a search returned: the id it is cited by, where it is from, and its text."""

    id: str
    source: str
    text: str

    @property
    def report(self) -> str:
        """The report it is from: ITC-FY25 for ITC-FY25#0412."""
        return self.id.split("#")[0]


def passages_in(result: str) -> list[Passage]:
    """Every passage in a search result's text, in order."""
    return [Passage(*found) for found in _PASSAGE.findall(result)]


def _lookup(result: str) -> Passage | None:
    """A fundamentals answer as evidence, cited by its `cite_as`; None if it is not one."""
    try:
        found = json.loads(result)
    except json.JSONDecodeError:
        return None
    if not isinstance(found, dict) or "cite_as" not in found:
        return None
    fields = cast(dict[str, object], found)
    return Passage(str(fields["cite_as"]), str(fields.get("source", "")), result)


@dataclass
class Toolbox:
    """`tools` is what the generalist is shown; calling the toolbox runs one: a specialist's
    through `route`, any other through `served`. Never raises for a bad call; a specialist that
    cannot be reached still fails the task."""

    served_tools: Sequence[ToolParam]
    served: Dispatch
    route: Router
    read_headline: Specialist[str, dict[str, JsonValue]]
    extract_figures: Specialist[FigureJob, Figures]
    evidence: dict[str, Passage] = field(default_factory=dict[str, Passage])

    @property
    def tools(self) -> list[ToolParam]:
        return [*self.served_tools, READ_HEADLINE_TOOL, EXTRACT_FIGURES_TOOL]

    def __call__(self, name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        own: dict[str, Callable[[dict[str, object]], str]] = {
            READ_HEADLINE_TOOL["name"]: self._headline,
            EXTRACT_FIGURES_TOOL["name"]: self._figures,
        }
        if name in own:
            try:
                return own[name](arguments), False
            except Invalid as refused:
                return f"Error: {refused}", True
        content, is_error = self.served(name, arguments)
        if name == SEARCH:
            self.evidence.update({found.id: found for found in passages_in(content)})
        if name == LOOKUP:
            looked_up = _lookup(content)
            if looked_up is not None:
                self.evidence[looked_up.id] = looked_up
        return content, is_error

    def _headline(self, arguments: dict[str, object]) -> str:
        headline = arguments.get("headline")
        if not isinstance(headline, str) or not headline.strip():
            raise Invalid("headline must be one line of text")
        return json.dumps(self.route(self.read_headline, headline.strip()))

    def _figures(self, arguments: dict[str, object]) -> str:
        ids, wanted = arguments.get("chunk_ids"), arguments.get("wanted")
        if not isinstance(ids, list) or not ids or not isinstance(wanted, str) or not wanted:
            raise Invalid("chunk_ids must list at least one passage id, and wanted must say what")
        unknown = [str(i) for i in cast(list[object], ids) if i not in self.evidence]
        if unknown:
            raise Invalid(f"no passage {', '.join(unknown)} from search_filings in this run")
        found: list[dict[str, JsonValue]] = []
        for chunk_id in cast(list[str], ids):
            passage = self.evidence[chunk_id]
            job = FigureJob(passage.text, passage.report, wanted)
            for figure in self.route(self.extract_figures, job).figures:
                found.append({**figure.model_dump(), "chunk_id": chunk_id})
        return json.dumps({"figures": found})
