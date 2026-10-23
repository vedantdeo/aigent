"""The analyst's `fundamentals` tool: reported figures looked up in the facts database by company,
financial year, period, basis and the figure's description, each cited like a passage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, cast

from anthropic.types import ToolParam

from aigent.facts.cfg_facts import FACTS_TICKERS, FACTS_YEARS
from aigent.facts.lookup import BASES, PERIODS, Basis, Lookup, NotFound, Period

# As schema fields these hold a model to what the database holds. A test holds them to it.
Company = Literal[
    "BAJAJ-AUTO", "HDFCBANK", "ITC", "POWERGRID", "RELIANCE", "SUNPHARMA", "TATAMOTORS", "WIPRO"
]
FinancialYear = Literal["2024", "2025", "2026"]

FUNDAMENTALS_TOOL: ToolParam = {
    "name": "fundamentals",
    "description": (
        "Look up a reported figure from a company's NSE XBRL results filings: revenue, expenses, "
        "profit, EPS, assets, liabilities, borrowings, cash flows, segment results and the rest of "
        "the results format, for a financial year, quarter, half or nine months, consolidated or "
        "standalone, as originally reported, in rupees (EPS in rupees per share). Say in `what` "
        "which figure, in plain words ('profit attributable to owners', 'current borrowings'); up "
        "to five matching facts come back, each with the label it is filed under, so check the "
        "label is the figure you meant. Balance-sheet items are as at the end of the period. "
        "Dividends per share, notes and anything outside the results format are not here: search "
        "the filings for those. Cite a fact by its `cite_as`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "enum": list(FACTS_TICKERS)},
            "year": {
                "type": "string",
                "enum": [str(y) for y in FACTS_YEARS],
                "description": "The financial year a report ends: 2025 is FY 2024-25.",
            },
            "what": {"type": "string", "description": "The figure, in plain words."},
            "period": {
                "type": "string",
                "enum": list(PERIODS),
                "description": (
                    "The whole year, one quarter (Q1 is April to June), the first half (H1) or "
                    "the first nine months (9M)."
                ),
            },
            "basis": {"type": "string", "enum": list(BASES)},
            "segment": {
                "type": ["string", "null"],
                "description": "A business segment's name for a segment figure, else null.",
            },
        },
        "required": ["company", "year", "what", "period", "basis", "segment"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class FundamentalsTool:
    """`fundamentals` over one `Lookup`, recording what it cites; a bad call or nothing found comes
    back as an error message, never an exception."""

    lookup: Lookup
    looked_up: list[str] = field(default_factory=list[str])

    def __call__(self, arguments: dict[str, object]) -> str:
        company = str(arguments.get("company", ""))
        year = str(arguments.get("year", ""))
        what = str(arguments.get("what", "")).strip()
        period = str(arguments.get("period", "year"))
        basis = str(arguments.get("basis", "consolidated"))
        segment = arguments.get("segment")
        refusal = _refusal(company, year, what, period, basis, segment)
        if refusal:
            return f"Error: {refusal}"
        try:
            found = self.lookup(
                company,
                int(year),
                what,
                cast(Basis, basis),
                cast(Period, period),
                None if segment is None else str(segment),
            )
        except NotFound as missing:
            return f"Error: {missing}"
        self.looked_up.extend(f.cite_as for f in found)
        return json.dumps({"facts": [f.as_json() for f in found]})


def _refusal(
    company: str, year: str, what: str, period: str, basis: str, segment: object
) -> str | None:
    if company not in FACTS_TICKERS:
        return f"no company {company!r}; one of {', '.join(FACTS_TICKERS)}"
    if not year.isdigit() or int(year) not in FACTS_YEARS:
        return f"no year {year!r}; one of {', '.join(map(str, FACTS_YEARS))}"
    if not what:
        return "say which figure in `what`"
    if period not in PERIODS:
        return f"no period {period!r}; one of {', '.join(PERIODS)}"
    if basis not in BASES:
        return f"no basis {basis!r}; one of {', '.join(BASES)}"
    if segment is not None and not isinstance(segment, str):
        return "`segment` is a name or null"
    return None
