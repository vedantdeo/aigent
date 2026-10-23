"""The generalist's tools in one place: a server's tools over MCP, and every specialist as a tool
whose calls go through the router. Passages a search returns, and those a specialist's answer
cites, are kept as evidence by id, as is each fact a fundamentals answer returns, by its `cite_as`,
so the critic can read what a claim rests on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from anthropic.types import ToolParam

from aigent.analyst.router import Router
from aigent.analyst.specialists import Invalid, Passage, passages_in
from aigent.llm.core import Dispatch

SEARCH, LOOKUP = "search_filings", "fundamentals"


@dataclass(frozen=True)
class Expert:
    """A specialist as one of the generalist's tools: the card it is shown, and `answer`, which
    turns a call's arguments into the tool's result, putting each job through the router, adding
    what its answer cites to the evidence, and raising `Invalid` for arguments it cannot use.
    `analyst.experts` lists them."""

    card: ToolParam
    answer: Callable[[dict[str, object], MutableMapping[str, Passage], Router], str]

    @property
    def name(self) -> str:
        return self.card["name"]


def _lookup(result: str) -> list[Passage]:
    """Each fact of a fundamentals answer as evidence, cited by its `cite_as`."""
    try:
        answer = json.loads(result)
    except json.JSONDecodeError:
        return []
    if not isinstance(answer, dict):
        return []
    facts = cast(dict[str, object], answer).get("facts")
    if not isinstance(facts, list):
        return []
    found: list[Passage] = []
    for fact in cast(list[object], facts):
        if isinstance(fact, dict) and "cite_as" in fact:
            fields = cast(dict[str, object], fact)
            text = json.dumps(fields)
            found.append(Passage(str(fields["cite_as"]), str(fields.get("source", "")), text))
    return found


@dataclass
class Toolbox:
    """`tools` is what the generalist is shown; calling the toolbox runs one: an expert's through
    `route`, any other through `served`. Never raises for a bad call; a specialist that cannot be
    reached still fails the task. A call repeating one that failed is refused unrun, with the first
    failure. ValueError if two tools share a name."""

    served_tools: Sequence[ToolParam]
    served: Dispatch
    route: Router
    experts: Sequence[Expert]
    evidence: dict[str, Passage] = field(default_factory=dict[str, Passage])
    failed: dict[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        names = [tool["name"] for tool in self.tools]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"two tools are called {', '.join(twice)}")

    @property
    def tools(self) -> list[ToolParam]:
        return [*self.served_tools, *(expert.card for expert in self.experts)]

    def __call__(self, name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        call = f"{name} {json.dumps(arguments, sort_keys=True, default=str)}"
        if call in self.failed:
            return (
                f"Error: this exact call already failed ({self.failed[call]}). Change its "
                "arguments or answer without it.",
                True,
            )
        content, is_error = self._run(name, arguments)
        if is_error:
            self.failed[call] = content.removeprefix("Error: ")
        return content, is_error

    def _run(self, name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        for expert in self.experts:
            if expert.name == name:
                try:
                    content = expert.answer(arguments, self.evidence, self.route)
                except Invalid as refused:
                    return f"Error: {refused}", True
                return content, False
        content, is_error = self.served(name, arguments)
        if name == SEARCH:
            self.evidence.update({found.id: found for found in passages_in(content)})
        if name == LOOKUP:
            self.evidence.update({found.id: found for found in _lookup(content)})
        return content, is_error
