"""The analyst's `fundamentals` and `compare_unverified` tools: verified figures looked up in the
facts database, each cited like a passage; unverified ones held back until the agent has worked the
figure out itself, then shown for comparison and never citable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, cast

from anthropic.types import ToolParam

from aigent.facts.cfg_facts import COMPARE_AGREE_SHARE, FACTS_TICKERS, FACTS_YEARS
from aigent.facts.lookup import BASES, PERIODS, Basis, Found, Lookup, NotFound, Period
from aigent.facts.store import CHECKED_TIERS

# As schema fields these hold a model to what the database holds. A test holds them to it.
Company = Literal[
    "BAJAJ-AUTO",
    "HDFCBANK",
    "ICICIBANK",
    "ITC",
    "POWERGRID",
    "RELIANCE",
    "SUNPHARMA",
    "TATAMOTORS",
    "WIPRO",
]
FinancialYear = Literal["2024", "2025", "2026"]

FUNDAMENTALS_TOOL: ToolParam = {
    "name": "fundamentals",
    "description": (
        "Look up a verified reported figure from a company's NSE XBRL results filings: revenue, "
        "expenses, profit, EPS, assets, liabilities, borrowings, cash flows, segment results and "
        "the rest of the results format, for a financial year, quarter, half or nine months, "
        "consolidated or standalone, as originally reported, in rupees (EPS in rupees per share). "
        "Say in `what` which figure, in plain words ('profit attributable to owners', 'current "
        "borrowings'). Only facts checked against the company's filed documents or by a person "
        "come back, up to five, each with the label it is filed under, so check the label is the "
        "figure you meant; cite one by its `cite_as`. Balance-sheet items are as at the end of the "
        "period. **When no verified fact comes back**, work the figure out yourself from the "
        "filings (search_filings, calculate) without guessing, then call compare_unverified with "
        "the answer's `lookup_id` and your figure to see any unverified entries it was held back "
        "from. Dividends per share, notes and anything outside the results format are not here: "
        "search the filings for those."
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


COMPARE_TOOL: ToolParam = {
    "name": "compare_unverified",
    "description": (
        "After fundamentals returned no verified fact and you have worked the figure out yourself "
        "from the filings, compare your figure with the unverified entries fundamentals held back "
        "for that lookup. Pass the answer's `lookup_id`, your figure in the same unit as the facts "
        "(full rupees; EPS in rupees per share; a ratio as a plain number) and, in `how`, where it "
        "came from (the passage ids you read, the scale printed, any arithmetic). Each entry comes "
        "back with `agrees`: within half a percent of your figure. **Never cite an unverified "
        "entry**: if one agrees, your figure stands, cited to the passages you read; if none "
        "agrees, re-read the filing before relying on yours, and say in the memo that the two "
        "differ. Example: compare_unverified(lookup_id='L2', my_value=485868100000, how='HDFCBANK "
        "Q1 FY24 results, interest earned 48,58,681 in lac, x 1e5')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "lookup_id": {"type": "string", "description": "The fundamentals answer's lookup_id."},
            "my_value": {"type": "number", "description": "Your figure, in the facts' unit."},
            "how": {"type": "string", "description": "Where your figure came from."},
        },
        "required": ["lookup_id", "my_value", "how"],
        "additionalProperties": False,
    },
    "strict": True,
}


def agrees(mine: float, held: float, share: float = COMPARE_AGREE_SHARE) -> bool:
    """Equal within `share` of the larger of the two."""
    return abs(mine - held) <= share * max(abs(mine), abs(held))


@dataclass
class FundamentalsTool:
    """`fundamentals` and `compare_unverified` over one `Lookup`: verified facts shown and recorded
    as cited, unverified ones held per lookup for `compare`. A bad call or nothing found comes back
    as an error message, never an exception."""

    lookup: Lookup
    looked_up: list[str] = field(default_factory=list[str])
    held: dict[str, list[Found]] = field(default_factory=dict[str, list[Found]])

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
        verified = [f for f in found if f.tier in CHECKED_TIERS]
        lookup_id = f"L{len(self.held) + 1}"
        self.held[lookup_id] = [f for f in found if f.tier not in CHECKED_TIERS]
        self.looked_up.extend(f.cite_as for f in verified)
        answer: dict[str, object] = {
            "facts": [f.as_json() for f in verified],
            "lookup_id": lookup_id,
            "unverified_held": len(self.held[lookup_id]),
        }
        if not verified:
            answer["note"] = (
                "No verified figure. Work it out from the filings yourself, then call "
                "compare_unverified with this lookup_id and your figure."
            )
        return json.dumps(answer)

    def compare(self, arguments: dict[str, object]) -> str:
        """`compare_unverified`: the held entries for a lookup, each marked agreeing or not, with
        no `cite_as`, so none can be cited."""
        lookup_id = str(arguments.get("lookup_id", ""))
        mine = arguments.get("my_value")
        how = str(arguments.get("how", "")).strip()
        if lookup_id not in self.held:
            known = ", ".join(self.held) or "none yet"
            return f"Error: no lookup {lookup_id!r}; lookups this session: {known}"
        if isinstance(mine, bool) or not isinstance(mine, int | float):
            return "Error: `my_value` is your figure as a number; work it out before comparing"
        if not how:
            return "Error: say in `how` where your figure came from"
        entries = [
            {
                "label": f.label,
                "value": f.value,
                "unit": f.unit,
                "period": f.as_json()["period"],
                "basis": f.basis,
                "segment": f.segment,
                "tier": f.tier,
                "agrees": agrees(float(mine), f.value),
            }
            for f in self.held[lookup_id]
        ]
        return json.dumps(
            {
                "lookup_id": lookup_id,
                "my_value": mine,
                "unverified": entries,
                "note": "Unverified: never cite these. Cite the passages your figure came from.",
            }
        )


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
