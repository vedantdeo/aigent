"""The generalist's tools in one place: a server's tools over MCP, and every specialist as a tool
whose calls go through the router. Passages a search returns are kept as evidence, so a later call
can name them by id.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from anthropic.types import ToolParam

from aigent.analyst.router import Router
from aigent.analyst.specialists import Invalid
from aigent.llm.core import Dispatch

SEARCH, LOOKUP = "search_filings", "fundamentals"
_PASSAGE = re.compile(r'<passage id="([^"]+)" source="([^"]*)">\n(.*?)\n</passage>', re.S)


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


@dataclass(frozen=True)
class Expert:
    """A specialist as one of the generalist's tools: the card it is shown, and `answer`, which
    turns a call's arguments into the tool's result, putting each job through the router and
    raising `Invalid` for arguments it cannot use. `analyst.experts` lists them."""

    card: ToolParam
    answer: Callable[[dict[str, object], Mapping[str, Passage], Router], str]

    @property
    def name(self) -> str:
        return self.card["name"]


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
    """`tools` is what the generalist is shown; calling the toolbox runs one: an expert's through
    `route`, any other through `served`. Never raises for a bad call; a specialist that cannot be
    reached still fails the task. ValueError if two tools share a name."""

    served_tools: Sequence[ToolParam]
    served: Dispatch
    route: Router
    experts: Sequence[Expert]
    evidence: dict[str, Passage] = field(default_factory=dict[str, Passage])

    def __post_init__(self) -> None:
        names = [tool["name"] for tool in self.tools]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"two tools are called {', '.join(twice)}")

    @property
    def tools(self) -> list[ToolParam]:
        return [*self.served_tools, *(expert.card for expert in self.experts)]

    def __call__(self, name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        for expert in self.experts:
            if expert.name == name:
                try:
                    return expert.answer(arguments, self.evidence, self.route), False
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
