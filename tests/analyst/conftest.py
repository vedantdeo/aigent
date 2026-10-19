"""A toolbox over scripted specialists, shared by the analyst's tests."""

from __future__ import annotations

from collections.abc import Callable

from aigent.analyst.experts import figures_expert
from aigent.analyst.router import Router
from aigent.analyst.specialists import FigureJob, Specialist, Told, check_figures
from aigent.analyst.toolbox import Toolbox
from aigent.retrieval.chunk import context_block

from ..conftest import PASSAGES

DIVIDEND = "RELIANCE-FY25#0001"  # "The Board recommended a dividend of Rs 5.50 per equity share."
FIGURE = '{"metric": "dividend", "value": 5.5, "unit": "INR", "year": 2025, "description": null}'


def told(text: str) -> Told:
    return Told(text, 0.01, 100, 10)


def never(job: object) -> Told:
    raise AssertionError("the generalist was asked")


def box(
    served: Callable[[str, dict[str, object]], tuple[str, bool]],
    figures: Callable[[FigureJob], Told] = lambda job: told('{"figures": [' + FIGURE + "]}"),
) -> Toolbox:
    return Toolbox(
        served_tools=[{"name": "search_filings", "input_schema": {"type": "object"}}],
        served=served,
        route=Router(price=lambda i, o: 0.0),
        experts=[
            figures_expert(Specialist("extract_figures", figures, never, check_figures)),
        ],
    )


def search(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
    assert name == "search_filings"
    return context_block([c for chunks in PASSAGES.values() for c in chunks]), False
