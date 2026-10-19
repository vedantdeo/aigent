"""The list of specialists the generalist is offered. Each is an `Expert`: the card that tells the
generalist what it is for, and how a call's arguments become jobs for its fine-tune. Adding one is
its card, its `…_expert` and a row in `BUILDERS`; nothing else names a specialist.
"""

from __future__ import annotations

import json
from collections.abc import Callable, MutableMapping

from anthropic.types import ToolParam

from aigent.analyst.cfg_analyst import GENERALIST_MODEL
from aigent.analyst.filings import COMPANIES, YEARS
from aigent.analyst.router import Router
from aigent.analyst.served import report_analyst
from aigent.analyst.specialists import FigureJob, Found, Invalid, Passage, Specialist
from aigent.analyst.toolbox import Expert
from aigent.llm.core import Dispatch, Llm

# A card's description is the generalist's whole knowledge of a specialist: what it is for, when
# to use it, and its limits, in that order.
EXTRACT_FIGURES_TOOL: ToolParam = {
    "name": "extract_figures",
    "description": (
        "Specialist for the figures in one company's annual report. Use it whenever a question "
        "needs figures from the filings, rather than searching and reading the passages "
        "yourself: give it the question exactly as you were asked it, the company's ticker and "
        "the financial year the report ends. It searches that report itself and returns each "
        "figure it needs: a metric, a value in full with any named scale applied, a unit, the "
        "financial year, and the id of the passage it is printed in, to cite. Limits: one "
        "company's report for one year a call, though a report also states the year before; no "
        "arithmetic across figures; a figure the report does not state is left out."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question, word for word."},
            "ticker": {"type": "string", "enum": [*COMPANIES], "description": "The company."},
            "year": {
                "type": "string",
                "enum": [*map(str, YEARS)],
                "description": "The financial year the report ends (2025 is FY 2024-25).",
            },
        },
        "required": ["question", "ticker", "year"],
        "additionalProperties": False,
    },
    "strict": True,
}


def figures_expert(specialist: Specialist[FigureJob, Found]) -> Expert:
    """`extract_figures` over `specialist`: one job for the question and the report named, its
    figures returned each with the passage it is printed in, and those passages added to the
    evidence so a claim can cite them."""

    def answer(
        arguments: dict[str, object], evidence: MutableMapping[str, Passage], route: Router
    ) -> str:
        question, ticker, year = (arguments.get(k) for k in ("question", "ticker", "year"))
        if not isinstance(question, str) or not question.strip():
            raise Invalid("question must be the question as you were asked it")
        if ticker not in COMPANIES:
            raise Invalid(f"ticker must be one of {', '.join(COMPANIES)}")
        if not isinstance(year, str) or not year.isdigit() or int(year) not in YEARS:
            raise Invalid(f"year must be one of {', '.join(map(str, YEARS))}")
        found = route(specialist, FigureJob(question.strip(), str(ticker), int(year)))
        evidence.update(found.passages)
        return json.dumps({"figures": [figure.model_dump() for figure in found.figures]})

    return Expert(EXTRACT_FIGURES_TOOL, answer)


# The client that serves each specialist's fine-tune, and how its expert is built from that
# client, the generalist, the served tools and the model it is asked as. Sorted by client.
BUILDERS: dict[str, Callable[[Llm, Llm, Dispatch, str | None], Expert]] = {
    "hostllm-analyst": lambda small, big, served, model: figures_expert(
        report_analyst(small, big, served, model)
    ),
}


def experts(
    big: Llm,
    served: Dispatch,
    small: Callable[[str], Llm] = Llm,
    model: str | None = GENERALIST_MODEL,
) -> list[Expert]:
    """Every specialist in `BUILDERS`, each on its own client's `Llm` from `small`, using the
    tools `served` runs, with `big`, asked as `model`, as the generalist they fall back to."""
    return [build(small(client), big, served, model) for client, build in BUILDERS.items()]
