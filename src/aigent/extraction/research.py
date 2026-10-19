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


BASES = ("consolidated", "standalone")


class SourcedFigure(Figure):
    """A figure, the basis it is reported on, and the passage that prints it. Its description may
    be the model's own words, where the per-passage task's is a quote."""

    description: str | None = Field(
        default=None,
        description=(
            "When metric is other, a few words, ten at most, naming what is counted or measured, "
            "in the passage's terms where they serve: 'retail outlets', 'vessels'. For a named "
            "metric, 'change' when the figure is its change, else null: only a named metric "
            "takes 'change', and it takes nothing else."
        ),
    )

    basis: str | None = Field(
        default=None,
        description=(
            "Only for a figure from the financial statements or segment information: "
            "consolidated or standalone, as the statement or its heading says, or for one "
            "segment's figure the segment's name copied from the passage. Null for anything "
            "else, such as a count, customers or business an initiative sourced."
        ),
    )
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
    "You answer a question about one Indian company's annual report with the figures it needs, "
    "found by searching that report. You have six turns: search, calculate, then answer.\n"
    "Searching. Query in the report's own words, naming the statement or table that would "
    "print the figure: 'Consolidated Statement of Profit and Loss', 'Revenue from operations', "
    "'segment results'. Search again with other words when the passages do not print every "
    "figure the question asks for, each year included; a report prints the previous year beside "
    "the current one, so both years usually come from one passage. When two searches find "
    "nothing, answer with no figures rather than guess.\n"
    "Reading. The question may also ask about other companies or other reports: answer only for "
    "the company and report named in <company> and <report>, and never search for another "
    "company; the rest of the question is not yours. Take figures on the basis the question names "
    "(consolidated, standalone or a segment) and, when it names none, consolidated. In a table, "
    "read each number by its row and its column heading: a column is a year only when its "
    "heading says so, and a segment table's columns are segments. When the question asks for "
    "something the report does not print, such as a ratio or a growth rate, return the figures "
    "it rests on and do not compute it.\n"
    "Answering. Return only the figures that answer the question, each from a passage a search "
    "returned, as JSON of the form "
    '{"figures": [{"metric": ..., "value": ..., "unit": ..., "year": ..., "description": ..., '
    '"basis": ..., "chunk_id": ...}]}, and nothing else. '
    + TOOL_RULE
    + " Each figure:\n"
    + _rules()
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
