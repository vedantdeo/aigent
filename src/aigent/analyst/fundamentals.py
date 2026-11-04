"""The analyst's `get_metric`, `fundamentals` and `compare_unverified` tools: concept names found
for the words of a question, a verified figure fetched by its exact concept, cited like a passage;
an unverified one held back until the agent has worked the figure out itself, then shown for
comparison and never citable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, cast

from anthropic.types import ToolParam

from aigent.facts.cfg_facts import (
    COMPANY_SCOPES,
    COMPARE_AGREE_SHARE,
    FACTS_LOOKUP_K,
    FACTS_TICKERS,
    FACTS_YEARS,
)
from aigent.facts.derived_concepts import kind, load_derived_concepts, ranked
from aigent.facts.lookup import (
    BASES,
    LENGTHS,
    QUARTERS,
    Basis,
    Found,
    Lookup,
    as_period,
)
from aigent.facts.store import CHECKED_TIERS, connect

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

GET_METRIC_TOOL: ToolParam = {
    "name": "get_metric",
    "description": (
        "Always first for a figure: turn its words into the exact names to fetch or compute it "
        "by, for one company. Returns up to five `filed` concepts the company files (fetch one "
        "with fundamentals, by its `concept`) and up to five `derived` metrics in its scope, "
        "built by a formula (plan one with plan_metric, by its `name`), each with its label and "
        "kind: `flow`, over a span, or `snapshot`, a balance as at a date. Read the labels and "
        "pick the one that is the figure asked; none may be. Example: get_metric(company="
        "'ICICIBANK', what='net interest income') returns the derived NetInterestIncome and "
        "filed concepts such as InterestEarned. If nothing fits, define a metric with "
        "define_metric or say the figure is not held."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "enum": list(FACTS_TICKERS)},
            "what": {"type": "string", "description": "The figure, in plain words."},
        },
        "required": ["company", "what"],
        "additionalProperties": False,
    },
    "strict": True,
}

FUNDAMENTALS_TOOL: ToolParam = {
    "name": "fundamentals",
    "description": (
        "Fetch one reported figure from a company's NSE XBRL results filings by its exact "
        "`concept` name, which get_metric or a plan_metric fetch list gives: never guess one. "
        "Name the time as a financial year and a quarter, meaning that quarter's end (year="
        "'2026', quarter='Q1' is 30 June 2025), and the period as the length of the span ending "
        "then: '3M' at any quarter, '6M' at Q2, '9M' at Q3, '12M' at Q4. A snapshot concept, a "
        "balance as at a date (Advances, Deposits, Assets), takes period null; a wrong "
        "combination is refused, saying which. Values are as originally reported, in rupees (EPS "
        "in rupees per share), consolidated or standalone. Only a fact checked against the "
        "company's filed documents or by a person comes back; cite it by its `cite_as`. **When no "
        "verified fact comes back**, work the figure out yourself from the filings without "
        "guessing, then call compare_unverified with the answer's `lookup_id` and your figure to "
        "see the unverified entry it was held back from. Example: fundamentals(company="
        "'ICICIBANK', concept='InterestEarned', year='2025', quarter='Q4', period='12M', "
        "basis='standalone', segment=null)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "enum": list(FACTS_TICKERS)},
            "concept": {"type": "string", "description": "The exact concept name, InterestEarned."},
            "year": {
                "type": "string",
                "enum": [str(y) for y in FACTS_YEARS],
                "description": "The time's financial year: 2025 is FY 2024-25.",
            },
            "quarter": {"type": "string", "enum": list(QUARTERS)},
            "period": {
                "anyOf": [{"type": "string", "enum": list(LENGTHS)}, {"type": "null"}],
                "description": "The span ending at the time, or null for a snapshot concept.",
            },
            "basis": {"type": "string", "enum": list(BASES)},
            "segment": {
                "type": ["string", "null"],
                "description": "A business segment's name for a segment figure, else null.",
            },
        },
        "required": ["company", "concept", "year", "quarter", "period", "basis", "segment"],
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
        concept = str(arguments.get("concept", "")).strip()
        quarter = str(arguments.get("quarter", ""))
        length = arguments.get("period")
        basis = str(arguments.get("basis", "consolidated"))
        segment = arguments.get("segment")
        refusal = _refusal(company, year, concept, quarter, length, basis, segment)
        if refusal:
            return f"Error: {refusal}"
        snapshot = self.lookup.snapshot(company, concept)
        if snapshot is None:
            return f"Error: {company} files no concept {concept!r}; find its name with get_metric"
        try:
            at_year, period = as_period(
                int(year), quarter, None if length is None else str(length), snapshot, concept
            )
            found = self.lookup.fact(
                company,
                concept,
                at_year,
                period,
                cast(Basis, basis),
                None if segment is None else str(segment),
            )
        except ValueError as refused:  # NotFound is one
            return f"Error: {refused}"
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

    def candidates(self, arguments: dict[str, object]) -> str:
        """`get_metric`: the filed concepts and the derived metrics closest to `what`."""
        company = str(arguments.get("company", ""))
        what = str(arguments.get("what", "")).strip()
        if company not in FACTS_TICKERS:
            return f"Error: no company {company!r}; one of {', '.join(FACTS_TICKERS)}"
        if not what:
            return "Error: say which figure in `what`"
        scope = COMPANY_SCOPES[company]
        conn = connect(self.lookup.db)
        try:
            held = load_derived_concepts(conn, scope)
            by_name = {m.name: m for m in held}
            derived = [
                {"name": m.name, "label": m.label, "kind": kind(conn, m, by_name)}
                for m in ranked(what, held)[:FACTS_LOOKUP_K]
            ]
        finally:
            conn.close()
        filed = [
            {"concept": c.concept, "label": c.label, "kind": "snapshot" if c.snapshot else "flow"}
            for c in self.lookup.candidates(company, what)
        ]
        return json.dumps(
            {
                "company": company,
                "scope": scope,
                "derived": derived,
                "filed": filed,
                "note": "Plan a derived one with plan_metric by its name; fetch a filed one with "
                "fundamentals by its concept.",
            }
        )

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
    company: str,
    year: str,
    concept: str,
    quarter: str,
    length: object,
    basis: str,
    segment: object,
) -> str | None:
    if company not in FACTS_TICKERS:
        return f"no company {company!r}; one of {', '.join(FACTS_TICKERS)}"
    if not year.isdigit() or int(year) not in FACTS_YEARS:
        return f"no year {year!r}; one of {', '.join(map(str, FACTS_YEARS))}"
    if not concept:
        return "name the concept in `concept`; find it with get_metric"
    if quarter not in QUARTERS:
        return f"no quarter {quarter!r}; one of {', '.join(QUARTERS)}"
    if length is not None and length not in LENGTHS:
        return f"no period {length!r}; one of {', '.join(LENGTHS)}, or null for a snapshot"
    if basis not in BASES:
        return f"no basis {basis!r}; one of {', '.join(BASES)}"
    if segment is not None and not isinstance(segment, str):
        return "`segment` is a name or null"
    return None
