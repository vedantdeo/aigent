"""The list of specialists the generalist is offered. Each is an `Expert`: the card that tells the
generalist what it is for, and how a call's arguments become jobs for its fine-tune. Adding one is
its card, its `…_expert` and a row in `BUILDERS`; nothing else names a specialist.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import cast

from anthropic.types import ToolParam
from pydantic import JsonValue

from aigent.analyst.router import Router
from aigent.analyst.served import headline_reader, report_analyst
from aigent.analyst.specialists import FigureJob, Invalid, Specialist
from aigent.analyst.toolbox import Expert, Passage
from aigent.config import GENERALIST_MODEL
from aigent.extraction.figures import Figures
from aigent.llm.core import Llm

# A card's description is the generalist's whole knowledge of a specialist: what it is for, when
# to use it, and its limits, in that order.
READ_HEADLINE_TOOL: ToolParam = {
    "name": "read_headline",
    "description": (
        "Specialist for market results headlines. Use it for any headline about an Indian "
        "listed company's results, rather than interpreting the headline yourself. It returns a "
        "record: the company as an NSE ticker, the metric, the quarter, the direction and the "
        "percentage change the headline states. Limits: one headline at a time; a company the "
        "directory does not list is an error."
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
        "Specialist for the figures in filing passages. Use it whenever you need a number from "
        "a passage search_filings returned, rather than reading the number yourself. It returns "
        "each figure the passages state: a metric, a value in full with any named scale applied, "
        "a unit and the financial year, checked against its passage and tagged with it. Limits: "
        "it reads each passage on its own and does no arithmetic across them."
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


def headline_expert(specialist: Specialist[str, dict[str, JsonValue]]) -> Expert:
    """`read_headline` over `specialist`: one headline, trimmed, to its record as JSON."""

    def answer(arguments: dict[str, object], evidence: Mapping[str, Passage], route: Router) -> str:
        headline = arguments.get("headline")
        if not isinstance(headline, str) or not headline.strip():
            raise Invalid("headline must be one line of text")
        return json.dumps(route(specialist, headline.strip()))

    return Expert(READ_HEADLINE_TOOL, answer)


def figures_expert(specialist: Specialist[FigureJob, Figures]) -> Expert:
    """`extract_figures` over `specialist`: one job per passage named, each figure returned tagged
    with the passage it came from."""

    def answer(arguments: dict[str, object], evidence: Mapping[str, Passage], route: Router) -> str:
        ids, wanted = arguments.get("chunk_ids"), arguments.get("wanted")
        if not isinstance(ids, list) or not ids or not isinstance(wanted, str) or not wanted:
            raise Invalid("chunk_ids must list at least one passage id, and wanted must say what")
        unknown = [str(i) for i in cast(list[object], ids) if i not in evidence]
        if unknown:
            raise Invalid(f"no passage {', '.join(unknown)} from search_filings in this run")
        found: list[dict[str, JsonValue]] = []
        for chunk_id in cast(list[str], ids):
            passage = evidence[chunk_id]
            job = FigureJob(passage.text, passage.report, wanted)
            for figure in route(specialist, job).figures:
                found.append({**figure.model_dump(), "chunk_id": chunk_id})
        return json.dumps({"figures": found})

    return Expert(EXTRACT_FIGURES_TOOL, answer)


# The client that serves each specialist's fine-tune, and how its expert is built from that
# client, the generalist and the model it is asked as. Sorted by client.
BUILDERS: dict[str, Callable[[Llm, Llm, str | None], Expert]] = {
    "hostllm-analyst": lambda small, big, model: figures_expert(report_analyst(small, big, model)),
    "hostllm-headline": lambda small, big, model: headline_expert(
        headline_reader(small, big, model=model)
    ),
}


def experts(
    big: Llm, small: Callable[[str], Llm] = Llm, model: str | None = GENERALIST_MODEL
) -> list[Expert]:
    """Every specialist in `BUILDERS`, each on its own client's `Llm` from `small`, with `big`,
    asked as `model`, as the generalist they fall back to."""
    return [build(small(client), big, model) for client, build in BUILDERS.items()]
