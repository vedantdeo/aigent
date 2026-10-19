"""The report analyst's job since 2026-10-19: one question about one company's report for one
financial year, answered with the figures it needs, found by searching that report.

Training rows and the served tool both build their prompt with `research_messages` and offer
`REPORT_SEARCH_TOOL` and calculate, so the fine-tune is taught exactly what it is later asked.
"""

from __future__ import annotations

import json

from anthropic.types import ToolParam
from pydantic import BaseModel, Field, ValidationError

from aigent.extraction.figures import TOOL_RULE, Figure

REPORT_SEARCH_TOOL: ToolParam = {
    "name": "search_report",
    "description": (
        "Search the annual report for passages that answer a query. Returns up to five "
        "passages, each tagged with the id to cite it by."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in the report's words."}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


class SourcedFigure(Figure):
    """A figure and the passage that prints it."""

    chunk_id: str = Field(description="The id of the passage the figure is printed in.")


class SourcedFigures(BaseModel):
    """What one call returns."""

    figures: list[SourcedFigure] = Field(
        description=(
            "Every figure the question needs that the report states, each with its passage's "
            "id; empty when the report does not state one."
        )
    )


def _rules() -> str:
    return "\n".join(f"- {name}: {f.description}" for name, f in SourcedFigure.model_fields.items())


SYSTEM = (
    "You answer a question about one Indian company's annual report with the figures it needs. "
    "Search the report with search_report, as often as the question needs, then return only the "
    "figures that answer it, each from a passage a search returned, as JSON of the form "
    '{"figures": [{"metric": ..., "value": ..., "unit": ..., "year": ..., "description": ..., '
    '"chunk_id": ...}]}, and nothing else. A report states the previous year beside the current '
    "one, so both years' figures may come from it. " + TOOL_RULE + " Each figure:\n" + _rules()
)


def research_messages(question: str, company: str, report: str) -> tuple[str, str]:
    """The system prompt and the user turn: the question, the company and the report it reads."""
    user = (
        f"<company>{company}</company>\n<report>{report}</report>\n<question>{question}</question>"
    )
    return SYSTEM, user


def parse_sourced(text: str) -> SourcedFigures | None:
    """The first JSON object in a reply as `SourcedFigures`; None when there is none or it does
    not fit."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return SourcedFigures.model_validate(json.loads(text[start : end + 1]))
    except (ValueError, ValidationError):
        return None
