"""The capstone's `extract_figures` as a schema, a prompt and a grader: the figures a report passage
states that answer what is wanted, as records.

The training rows, the eval and the tool all build their prompt with `figure_messages`, so the
fine-tune is taught exactly what it is later asked. The model applies scales and percentages by
calling `calculate`; underhood's `tooldata` writes the conversations that teach it to.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue, ValidationError

from aigent.evals.dataset import Case
from aigent.evals.grade import Outcome, Score
from aigent.retrieval.chunk import squeeze
from aigent.tools import CALCULATOR_TOOL

DATASET = Path(__file__).resolve().parents[3] / "evals" / "datasets" / "figures.jsonl"
CURRENCIES = ("INR", "USD", "AUD", "EUR", "GBP")
CHANGE = "change"
# The metrics a percentage may be, besides a change in any metric.
RATE_METRICS = ("margin", "roe", "roce", "roa", "market_share")
_RUPEE = {"rs", "rs.", "₹", "inr", "rupee", "rupees"}
_WORD = re.compile(r"[A-Za-z₹%]+\.?")

# Tie-break order, most specific first: the order is the data, so do not sort.
METRICS = (
    "margin",
    "roe",
    "roce",
    "roa",
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
            "margin is any percentage of revenue (EBITDA margin is margin); roe, roce and roa are "
            "the returns on equity, capital employed and assets; eps and dividend are "
            "per share; profit is profit after tax, or before tax when no other is stated; "
            "revenue is revenue from operations, sales or turnover; debt is what the company owes, "
            "its borrowings and the loan facilities it took, not the loans and advances it makes; "
            "volume is units sold or produced, tonnes or capacity; customers "
            "includes subscribers and users. A change in a named metric, a rise or a fall, as a "
            "rate or an amount, keeps that metric with description 'change'. other is any "
            "figure the rest do not name."
        )
    )
    value: float = Field(
        description=(
            "The number in full, with any scale word applied: 'Rs 1,250 crores' is "
            "12500000000, '3 lakh' is 300000, '4.5 million' is 4500000, '12,34,56,789' is "
            "123456789; a percentage as a fraction, '50%' is 0.5 and '7.25%' is 0.0725; 'over "
            "80' is 80, '60+' is 60, 'eight' is 8. A table heading such as '(Rs in crore)' "
            "scales every number under it. Negative when printed so, with a minus sign or in "
            "parentheses, and for a change that is a fall: 'declined 4%' is -0.04; otherwise a "
            "fall stated in words stays positive. Always a plain number, never "
            "an expression such as 10**6: an expression goes to calculate, its result here."
        )
    )
    unit: str | None = Field(
        description=(
            "For money, the currency code: " + ", ".join(CURRENCIES) + ". The rupee sign may "
            "appear as C, E, H, I, J, K or ` in this text. Money with no currency shown is INR, "
            "the report being Indian; an exchange rate beside a subsidiary's amounts is not "
            "their currency. Null for anything that is not money, a percentage included."
        )
    )
    year: int | None = Field(
        description=(
            "The Indian financial year (April to March) the figure belongs to, as the year it "
            "ends: 2025 for FY 2024-25, for as at 31 March 2025, or for an event on 20 June "
            "2024. Read the report's year from source: a figure the passage states as current, "
            "or 'during the year', takes it, so a headcount in a 2023-24 report is 2024, and "
            "'previous year' is the year before. A bare calendar year the report presents as "
            "current takes the report's year: an award won in 2023, in a 2023-24 report, is "
            "2024. Null only for a cumulative, 'till date' or 'since inception' total, a "
            "duration such as a three-year term, a fixed term or specification such as the face "
            "value of a share or a vehicle's seating capacity, or a year that only names "
            "something such as Vision 2030."
        )
    )
    description: str | None = Field(
        default=None,
        description=(
            "When metric is other, the word or few words of the passage that name what is "
            "counted or measured, copied verbatim: 'branches', 'retail outlets', 'vessels'. "
            "For a named metric, 'change' when the figure is its change, else null."
        ),
    )


class Figures(BaseModel):
    """What one call returns."""

    figures: list[Figure] = Field(
        description=(
            "Every figure in the passage that answers what is wanted, in the order it appears, "
            "a rate of change beside the amount it describes included; empty when none does. On "
            "a highlights page a number usually sits on the line above its label."
        )
    )


def _rules() -> str:
    lines = [f"- {name}: {f.description}" for name, f in Figure.model_fields.items()]
    return "\n".join(lines)


TOOL_RULE = (
    "Do no arithmetic yourself. When an amount carries a scale (thousand, lakh, crore, million, "
    "billion), beside it or in a table heading, or is a percentage, call calculate with the "
    "number as printed times its scale, or over 100, and copy the result into value. Every "
    "percentage takes a call: '18%' is calculate('18 / 100') and value 0.18, never 18. A number "
    "already printed in full needs no call."
)

SYSTEM = (
    "You extract figures from a passage of an Indian company's annual report. Return only the "
    "figures that answer what is wanted, as JSON of the form "
    '{"figures": [{"metric": ..., "value": ..., "unit": ..., "year": ..., "description": ...}]}, '
    "and nothing else. " + TOOL_RULE + " Each figure:\n" + _rules()
)

# The calculate tool as Hugging Face chat templates take it; aigent's own definition, re-shaped.
TOOL: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": CALCULATOR_TOOL["name"],
        "description": CALCULATOR_TOOL.get("description", ""),
        "parameters": json.loads(json.dumps(CALCULATOR_TOOL["input_schema"])),
    },
}


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
    """A description verbatim from the passage iff the metric is other; on a named metric,
    none, or exactly `CHANGE` for a change in it."""
    if figure.metric != "other":
        return figure.description in (None, CHANGE)
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


class Labelled(Figure):
    """A figure as the teacher returns it: with the calculate call behind its value, if any."""

    call: str | None = Field(
        description=(
            "The calculate expression behind value: the number as printed times its scale, "
            "'1,250 crore' is '1250 * 10**7' and '2.5' under a '(Rs in lakh)' heading is "
            "'2.5 * 10**5', or over 100 for a percentage, '7.25%' is '7.25 / 100'. Null when the "
            "number is printed in full."
        )
    )


_PRINTED = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMBER_WORDS = {
    name: i
    for i, name in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
}


def _printed(passage: str) -> set[float]:
    """Every number printed in a passage, commas dropped, plus the small ones written as words."""
    numbers = {float(n.replace(",", "")) for n in _PRINTED.findall(passage)}
    words = {float(v) for w, v in _NUMBER_WORDS.items() if re.search(rf"\b{w}\b", passage, re.I)}
    return numbers | words
