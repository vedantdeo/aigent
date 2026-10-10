"""The capstone's `extract_figures` as a schema, a prompt and a grader: the figures a report passage
states that answer what is wanted, as records.

The training rows, the eval and the tool all build their prompt with `figure_messages`, so the
fine-tune is taught exactly what it is later asked.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from aigent.evals.dataset import Case
from aigent.evals.grade import Outcome, Score
from aigent.retrieval.chunk import squeeze

DATASET = Path(__file__).resolve().parents[3] / "evals" / "datasets" / "figures.jsonl"
CURRENCIES = ("INR", "USD", "AUD", "EUR", "GBP")
_RUPEE = {"rs", "rs.", "₹", "inr", "rupee", "rupees"}
_WORD = re.compile(r"[A-Za-z₹%]+\.?")

# Tie-break order, most specific first: the order is the data, so do not sort.
METRICS = (
    "margin",
    "eps",
    "dividend",
    "ebitda",
    "profit",
    "revenue",
    "cash_flow",
    "capex",
    "debt",
    "market_share",
    "volume",
    "customers",
    "employees",
    "other",
)


class Figure(BaseModel):
    """One figure. Descriptions are the prompt, so they stay short."""

    metric: str = Field(
        description=(
            "Exactly one of: " + ", ".join(METRICS) + ". The first that fits, in that order: "
            "margin is any percentage of revenue (EBITDA margin is margin); eps and dividend are "
            "per share; profit is profit after tax, or before tax when no other is stated; "
            "revenue is revenue from operations, sales or turnover; debt includes borrowings and "
            "loan facilities; volume is units sold or produced, tonnes or capacity; customers "
            "includes subscribers and users. other is any figure the rest do not name."
        )
    )
    value: float = Field(
        description=(
            "The number in full, with any scale word applied: 'Rs 1,250 crores' is "
            "12500000000, '3 lakh' is 300000, '4.5 million' is 4500000, '12,34,56,789' is "
            "123456789; a percentage as a fraction, '50%' is 0.5 and '7.25%' is 0.0725; 'over "
            "80' is 80, '60+' is 60, 'eight' is 8."
        )
    )
    unit: str | None = Field(
        description=(
            "For money, the currency code: " + ", ".join(CURRENCIES) + ". The rupee sign may "
            "appear as C, I or ` in this text. Null for anything else, a percentage included."
        )
    )
    year: int | None = Field(
        description=(
            "The year the figure's period ends, when the passage states or dates it: 2025 for "
            "FY 2024-25, for 'during the year' in a 2024-25 report, or as at 31 March 2025; "
            "2024 for calendar 2024. Null for 'till date', 'since inception' or no period."
        )
    )
    description: str | None = Field(
        default=None,
        description=(
            "When metric is other, the word or few words of the passage that name what is "
            "counted or measured, copied verbatim: 'branches', 'retail outlets', 'vessels'. "
            "Null for every other metric."
        ),
    )


class Figures(BaseModel):
    """What one call returns."""

    figures: list[Figure] = Field(
        description=(
            "Every figure in the passage that answers what is wanted, in the order it appears; "
            "empty when none does."
        )
    )


def _rules() -> str:
    lines = [f"- {name}: {f.description}" for name, f in Figure.model_fields.items()]
    return "\n".join(lines)


SYSTEM = (
    "You extract figures from a passage of an Indian company's annual report. Return only the "
    "figures that answer what is wanted, as JSON of the form "
    '{"figures": [{"metric": ..., "value": ..., "unit": ..., "year": ..., "description": ...}]}, '
    "and nothing else. "
    "Each figure:\n" + _rules()
)


def figure_messages(passage: str, source: str, wanted: str) -> tuple[str, str]:
    """The system prompt and the user turn for one passage."""
    user = f"<source>{source}</source>\n<passage>{passage}</passage>\n<wanted>{wanted}</wanted>"
    return SYSTEM, user


def parse_figures(text: str) -> Figures | None:
    """The first JSON object in a reply as `Figures`; None when there is none or it does not fit."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return Figures.model_validate(json.loads(text[start : end + 1]))
    except (ValueError, ValidationError):
        return None


def unit_key(unit: str | None) -> str | None:
    """A unit as the grader compares it: a currency code, None, or the text as given."""
    if unit is None or not unit.strip():
        return None
    words = [w.lower().rstrip(".") for w in _WORD.findall(unit)]
    currency = next((c for c in CURRENCIES if c.lower() in words), None)
    if currency is None and any(w in _RUPEE for w in words):
        currency = "INR"
    return currency or unit.strip()


def _same(want: Figure, got: Figure) -> bool:
    return (
        want.metric == got.metric
        and math.isclose(want.value, got.value, rel_tol=1e-9, abs_tol=1e-9)
        and unit_key(want.unit) == unit_key(got.unit)
        and want.year == got.year
    )


def _described(figure: Figure, passage: object) -> bool:
    """A description iff the metric is other, and verbatim in the passage when there is one."""
    if figure.metric != "other":
        return figure.description is None
    if not figure.description:
        return False
    return not isinstance(passage, str) or squeeze(figure.description) in squeeze(passage)


def figures_match(case: Case, outcome: Outcome) -> Score:
    """Every expected figure found once on metric, value, unit and year, nothing extra, and
    every description where the metric needs one; verbatim when `case.input` carries the passage."""
    try:
        want = Figures.model_validate({"figures": case.expected.get("figures")}).figures
        got = Figures.model_validate({"figures": outcome.output.get("figures")}).figures
    except ValidationError as error:
        return Score(False, f"not figures: {error.errors()[0]['msg']}")
    left = list(got)
    missed: list[Figure] = []
    for figure in want:
        hit = next((g for g in left if _same(figure, g)), None)
        if hit is None:
            missed.append(figure)
        else:
            left.remove(hit)
    found = len(want) - len(missed)
    undescribed = [f for f in got if not _described(f, case.input.get("passage"))]
    detail = "; ".join(
        [f"missed {f.metric} {f.value:g} {f.unit} {f.year}" for f in missed]
        + [f"extra {f.metric} {f.value:g} {f.unit} {f.year}" for f in left]
        + [f"description wrong for {f.metric} {f.value:g}" for f in undescribed]
    )
    parts = {"all_found": not missed, "no_extra": not left, "described": not undescribed}
    recall = found / len(want) if want else 1.0
    return Score(all(parts.values()), detail, parts=parts, value=recall)
