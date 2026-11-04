"""The agent's metric tools, over one question: `plan_metric` names the facts a metric needs (in
the company's scope, through any derived concepts it builds on), `fundamentals` fetches them, and
`compute_metric` checks each against the plan and runs the library formulas on their stored
values, so the model never types a figure or a formula.
`define_metric` adds a metric built on a library formula, `propose_formula` asks a person for a new
formula; both are written at once as `-ai`. Once the critic supports a claim citing a computed
figure, `write_back` stores it under the metric's name: `derived` when its derived concept is
checked by a person, else `derived-ai`."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import cast

from anthropic.types import ToolParam

from aigent.analyst.graph import Analysed
from aigent.analyst.specialists import Passage
from aigent.facts.cfg_facts import COMPANY_SCOPES, FACTS_TICKERS, SCOPES
from aigent.facts.derived_concepts import (
    ATS,
    At,
    Binding,
    DerivedConcept,
    Fetch,
    Step,
    When,
    available,
    bindings_json,
    bindings_refusal,
    comparing,
    evaluate,
    kind,
    leaves,
    load_derived_concepts,
    name_refusal,
    plan,
)
from aigent.facts.formulas import FORMULAS
from aigent.facts.lookup import BASES, LENGTHS, QUARTERS, Period, as_period, as_time, instants, span
from aigent.facts.store import (
    CHECKED_TIERS,
    DerivedRow,
    Writer,
    concept_check,
    normal_scope,
    scope_refusal,
    write_derived,
    write_derived_concept,
    write_formula_proposal,
)
from aigent.llm.core import Dispatch

PLAN, COMPUTE, DEFINE, PROPOSE = "plan_metric", "compute_metric", "define_metric", "propose_formula"
_LIBRARY = "; ".join(
    f"{f.name}({', '.join(f.variables)}) = {f.description}" for f in FORMULAS.values()
)
_SCOPES = "; ".join(
    f"{scope} for {', '.join(t for t, s in COMPANY_SCOPES.items() if s == scope)}"
    for scope in SCOPES
)

PLAN_TOOL: ToolParam = {
    "name": PLAN,
    "description": (
        "Plan a derived metric get_metric found (a growth rate, a ratio, a margin) by its exact "
        "name: the exact facts it needs for one company and basis. Name "
        "the time as a financial year and a quarter, meaning that quarter's end (year='2026', "
        "quarter='Q1' is 30 June 2025), and the period as the length of the span ending then: "
        "'3M' at any quarter, '6M' at Q2, '9M' at Q3, '12M' at Q4. A snapshot metric, every "
        "input a balance as at a date (a credit-deposit ratio), has no period: pass null. A "
        "metric that compares two times, such as a growth rate, also needs previous_year, "
        "previous_quarter and previous_period, the time and period it is compared with: always "
        "pass them, as nothing is assumed; the previous period is as long and ends before the "
        "current one begins. A metric at one time takes null for all three. A wrong combination "
        "is refused, saying which. Returns a plan_id, the metric's kind (flow, snapshot or "
        "mixed) and library formula, and a fetch list: call fundamentals once per entry with "
        "exactly the arguments given, then compute_metric with the plan_id and each fact's "
        "cite_as. A name not held in the company's scope is refused: find one with get_metric or "
        "define one with define_metric. Examples: net profit growth over the quarter before, "
        "plan_metric(name='ProfitLossForThePeriodGrowth', company='ICICIBANK', year='2026', "
        "quarter='Q1', period='3M', "
        "previous_year='2025', previous_quarter='Q4', previous_period='3M', basis='standalone'); "
        "a balance-sheet ratio at the year's end, plan_metric(name='CreditDepositRatio', "
        "company='HDFCBANK', year='2025', quarter='Q4', period=null, previous_year=null, "
        "previous_quarter=null, previous_period=null, basis='consolidated')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The metric's exact name, from get_metric."},
            "company": {"type": "string", "enum": list(FACTS_TICKERS)},
            "year": {
                "type": "string",
                "description": "The financial year of the time: 2025 is FY 2024-25.",
            },
            "quarter": {"type": "string", "enum": list(QUARTERS)},
            "period": {
                "anyOf": [{"type": "string", "enum": list(LENGTHS)}, {"type": "null"}],
                "description": "The span ending at the time, or null for a snapshot metric.",
            },
            "previous_year": {
                "type": ["string", "null"],
                "description": "The compared time's financial year, or null at one time.",
            },
            "previous_quarter": {
                "anyOf": [{"type": "string", "enum": list(QUARTERS)}, {"type": "null"}]
            },
            "previous_period": {
                "anyOf": [{"type": "string", "enum": list(LENGTHS)}, {"type": "null"}]
            },
            "basis": {"type": "string", "enum": list(BASES)},
        },
        "required": [
            "name",
            "company",
            "year",
            "quarter",
            "period",
            "previous_year",
            "previous_quarter",
            "previous_period",
            "basis",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}

COMPUTE_TOOL: ToolParam = {
    "name": COMPUTE,
    "description": (
        "Step 3: compute a planned metric from the facts fundamentals returned for its fetch "
        "list. Pass the plan_id and, per fetch entry, its variable as listed (a path such as "
        "'denominator.first' when a metric builds on another) and the cite_as you fetched. "
        "Each cite must be the planned fact, verified; the values are taken from the facts. "
        "Returns the figure with a cite_as of its own (DERIVED:D1, …): cite that for it. A "
        "growth rate or ratio is a plain number, 0.1577 for 15.77%."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string"},
            "inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"variable": {"type": "string"}, "cite_as": {"type": "string"}},
                    "required": ["variable", "cite_as"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plan_id", "inputs"],
        "additionalProperties": False,
    },
    "strict": True,
}

DEFINE_TOOL: ToolParam = {
    "name": DEFINE,
    "description": (
        "Define a metric plan_metric did not find, built on a library formula: "
        f"{_LIBRARY}. Name it in CamelCase with no company or period (CostToIncomeRatio). Give "
        "its scope, the XBRL taxonomies it holds for: "
        f"{_SCOPES}; several joined by commas, or * for all, when every scope named files the "
        "concepts it uses. Bind each formula variable to a concept those scopes file (the names "
        "fundamentals returns, such as InterestEarned) or to a metric already defined in it, at "
        "'current' or 'previous': a metric reading 'previous' compares two periods, both named "
        "when it is planned (growth: current at 'current', previous at 'previous'), and cannot be "
        "an input itself. Name no period in it. It is "
        "saved, as a derived concept, unchecked for a person to review; then plan it. If no "
        "library formula fits even through other metrics, call propose_formula instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "scope": {
                "type": "string",
                "description": "One scope, several joined by commas, or * for every scope.",
            },
            "label": {"type": "string", "description": "What it is, in words."},
            "formula": {"type": "string", "enum": list(FORMULAS)},
            "bindings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string"},
                        "concept": {"type": "string"},
                        "at": {"type": "string", "enum": list(ATS)},
                    },
                    "required": ["variable", "concept", "at"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["name", "scope", "label", "formula", "bindings"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROPOSE_TOOL: ToolParam = {
    "name": PROPOSE,
    "description": (
        "When no library formula can express a figure, propose one for a person to add to the "
        "library: a name, the expression over named variables, and why it is needed. Nothing can "
        "be computed with it until a person adds it; say so in your answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "expression": {"type": "string"},
            "variables": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "expression", "variables"],
        "additionalProperties": False,
    },
    "strict": True,
}

METRIC_TOOLS: list[ToolParam] = [PLAN_TOOL, COMPUTE_TOOL, DEFINE_TOOL, PROPOSE_TOOL]


@dataclass(frozen=True)
class Plan:
    metric: DerivedConcept
    company: str
    year: int
    period: Period
    basis: str
    step: Step
    previous: When | None = None
    snapshot: bool = False

    @property
    def fetches(self) -> list[Fetch]:
        return self.step.fetches()


@dataclass(frozen=True)
class Computed:
    cite_as: str
    plan: Plan
    value: float
    unit: str
    inputs: tuple[tuple[str, str], ...]  # (variable, the fact's cite_as)


@dataclass
class MetricTools:
    """The four tools over the facts database `conn`, the question's `evidence`, and who writes.
    Every refusal is an error result naming what to fix."""

    conn: sqlite3.Connection
    evidence: MutableMapping[str, Passage]
    writer: Writer
    question: str
    plans: dict[str, Plan] = field(default_factory=dict[str, Plan])
    made: dict[str, Computed] = field(default_factory=dict[str, Computed])

    def __call__(self, name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        tools = {PLAN: self.plan, COMPUTE: self.compute, DEFINE: self.define, PROPOSE: self.propose}
        try:
            return json.dumps(tools[name](arguments)), False
        except ValueError as bad:
            return f"Error: {bad}", True

    def plan(self, arguments: dict[str, object]) -> dict[str, object]:
        name, company = str(arguments.get("name", "")).strip(), str(arguments.get("company", ""))
        year, quarter = str(arguments.get("year", "")), str(arguments.get("quarter", ""))
        length = arguments.get("period")
        basis = str(arguments.get("basis", ""))
        was = [arguments.get(f"previous_{k}") for k in ("year", "quarter", "period")]
        if not name or company not in FACTS_TICKERS or not year.isdigit():
            raise ValueError("give the name, a company of the database and the year as digits")
        if quarter not in QUARTERS or basis not in BASES:
            raise ValueError(f"quarter is one of {', '.join(QUARTERS)}; basis one of {BASES}")
        if length is not None and length not in LENGTHS:
            raise ValueError(f"period is one of {', '.join(LENGTHS)}, or null for a snapshot")
        compared = was[0] is not None or was[1] is not None
        if compared and (not str(was[0]).isdigit() or was[1] not in QUARTERS):
            raise ValueError("give previous_year as digits and previous_quarter, or null for both")
        if was[2] is not None and (not compared or was[2] not in LENGTHS):
            raise ValueError("previous_period is one of 3M, 6M, 9M, 12M, with a previous time")
        scope = COMPANY_SCOPES[company]
        by_name = {m.name: m for m in load_derived_concepts(self.conn, scope)}
        metric = by_name.get(name)
        if metric is None:
            raise ValueError(
                f"no metric {name!r} in the {scope} scope; find one with get_metric or define "
                f"one with define_metric on the library: {_LIBRARY}"
            )
        shape = kind(self.conn, metric, by_name)
        snapshot = shape == "snapshot"
        at = as_period(int(year), quarter, cast(str | None, length), snapshot, metric.name)
        previous = (
            as_period(
                int(str(was[0])), str(was[1]), cast(str | None, was[2]), snapshot, metric.name
            )
            if compared
            else None
        )
        if previous is not None and was[2] != length:
            raise ValueError("previous_period is as long as period; compare like with like")
        balances = instants(self.conn, sorted(leaves(metric, by_name)))
        step = plan(metric, *at, by_name, previous, balances=balances)
        plan_id = f"P{len(self.plans) + 1}"
        self.plans[plan_id] = Plan(metric, company, *at, basis, step, previous, snapshot)
        return {
            "plan_id": plan_id,
            "metric": metric.name,
            "scope": scope,
            "label": metric.label,
            "kind": shape,
            "periods": _periods(*at, previous, snapshot),
            "tier": metric.tier,
            "formula": f"{metric.formula}: {FORMULAS[metric.formula].description}",
            "fetch": [
                {
                    "variable": f.variable,
                    "fundamentals": {
                        "company": company,
                        "concept": f.concept,
                        "year": str(f.year),
                        "quarter": as_time(f.period)[0],
                        "period": None if f.concept in balances else as_time(f.period)[1],
                        "basis": basis,
                        "segment": None,
                    },
                    "cite_as": f.cite_as(company, basis),
                }
                for f in step.fetches()
            ],
        }

    def compute(self, arguments: dict[str, object]) -> dict[str, object]:
        plan_id = str(arguments.get("plan_id", ""))
        planned = self.plans.get(plan_id)
        if planned is None:
            raise ValueError(
                f"no plan {plan_id!r}; plans so far: {', '.join(self.plans) or 'none'}"
            )
        given = _pairs(arguments.get("inputs"), "variable", "cite_as")
        wanted = {f.variable: f.cite_as(planned.company, planned.basis) for f in planned.fetches}
        if sorted(given) != sorted(wanted):
            raise ValueError(f"give one cite_as for each of {', '.join(wanted)}")
        facts: dict[str, dict[str, object]] = {}
        for variable, cite in given.items():
            if cite != wanted[variable]:
                raise ValueError(f"{variable} is {wanted[variable]}, not {cite}")
            facts[variable] = self._fact(cite)
        formula = FORMULAS[planned.metric.formula]
        value, unit = evaluate(
            planned.step,
            {v: (float(cast(float, f["value"])), str(f["unit"])) for v, f in facts.items()},
        )
        cite_as = f"DERIVED:D{len(self.made) + 1}"
        inputs = tuple((f.variable, wanted[f.variable]) for f in planned.fetches)
        self.made[cite_as] = Computed(cite_as, planned, value, unit, inputs)
        shown: dict[str, object] = {
            "cite_as": cite_as,
            "metric": planned.metric.name,
            "label": planned.metric.label,
            "value": value,
            "unit": unit,
            "trusted": planned.metric.tier == "derived-concept",
            "scope": planned.metric.scope,
            "formula": f"{formula.name}: {formula.description}",
            "company": planned.company,
            "periods": _periods(planned.year, planned.period, planned.previous, planned.snapshot),
            "basis": planned.basis,
            "inputs": [{"variable": v, **facts[v]} for v, _ in inputs],
        }
        self.evidence[cite_as] = Passage(
            cite_as, "computed in this conversation", json.dumps(shown)
        )
        return shown

    def define(self, arguments: dict[str, object]) -> dict[str, object]:
        name, label = str(arguments.get("name", "")), str(arguments.get("label", "")).strip()
        formula, scope = str(arguments.get("formula", "")), str(arguments.get("scope", ""))
        if why := scope_refusal(scope):
            raise ValueError(why)
        scope = normal_scope(scope)
        if why := name_refusal(name):
            raise ValueError(why)
        if not label:
            raise ValueError("give the metric a label in words")
        bindings: dict[str, Binding] = {}
        for entry in cast(list[object], arguments.get("bindings") or []):
            row = cast(dict[str, object], entry) if isinstance(entry, dict) else {}
            variable = str(row.get("variable", ""))
            if variable in bindings:
                raise ValueError(f"{variable} is bound twice")
            bindings[variable] = Binding(str(row.get("concept", "")), cast(At, row.get("at")))
        if why := bindings_refusal(
            formula, bindings, available(self.conn, scope), name, comparing(self.conn, scope)
        ):
            raise ValueError(why)
        with self.conn:
            written = write_derived_concept(
                self.conn,
                name,
                scope,
                label,
                formula,
                bindings_json(bindings),
                self.question,
                self.writer,
            )
        if not written:
            raise ValueError(
                f"{name} is already defined in a scope overlapping {scope}; plan it by name"
            )
        return {
            "defined": name,
            "scope": scope,
            "tier": "derived-concept-ai",
            "note": "Saved unchecked; now plan it.",
        }

    def propose(self, arguments: dict[str, object]) -> dict[str, object]:
        name = str(arguments.get("name", "")).strip()
        expression = str(arguments.get("expression", "")).strip()
        variables = [str(v) for v in cast(list[object], arguments.get("variables") or [])]
        if not name or not expression or not variables:
            raise ValueError("give the formula's name, its expression and its variables")
        with self.conn:
            formula_id, new = write_formula_proposal(
                self.conn, name, expression, variables, self.question, self.writer
            )
        return {
            "proposed": formula_id,
            "tier": "formula-ai",
            "new": new,
            "note": "A person adds it to the library; nothing can be computed with it until then.",
        }

    def _fact(self, cite: str) -> dict[str, object]:
        found = self.evidence.get(cite)
        if found is None:
            raise ValueError(f"{cite} was not returned by fundamentals here; fetch it first")
        fact = cast(dict[str, object], json.loads(found.text))
        if fact.get("tier") not in CHECKED_TIERS:
            raise ValueError(f"{cite} is not verified")
        return fact


def _periods(year: int, period: Period, previous: When | None, snapshot: bool = False) -> str:
    """The time a figure is at, or the two it compares: "Q1 FY2026 over Q4 FY2025"; a snapshot
    is "as at the end of Q4 FY2025"."""
    at = _named(year, period)
    told = at if previous is None else f"{at} over {_named(*previous)}"
    return f"as at the end of {told}" if snapshot else told


def _named(year: int, period: Period) -> str:
    return f"FY{year}" if period == "year" else f"{period} FY{year}"


def _pairs(given: object, key: str, value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in cast(list[object], given) if isinstance(given, list) else []:
        row = cast(dict[str, object], entry) if isinstance(entry, dict) else {}
        if str(row.get(key, "")) in out:
            raise ValueError(f"{row.get(key)} is given twice")
        out[str(row.get(key, ""))] = str(row.get(value, ""))
    return out


def chained(tools: MetricTools, rest: Dispatch) -> Dispatch:
    """A dispatcher sending the metric tools to `tools` and every other tool to `rest`."""

    def call(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        return (
            tools(name, arguments)
            if name in (PLAN, COMPUTE, DEFINE, PROPOSE)
            else rest(name, arguments)
        )

    return call


def write_back(
    conn: sqlite3.Connection, done: Analysed, tools: MetricTools, writer: Writer
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Writes each computed figure a supported claim cites, once, and returns what was written and
    what was skipped with its reason."""
    if done.memo is None:
        return [], []
    supported = [
        cite
        for claim in done.memo.claims
        if (judged := done.verdicts.get(claim.id)) and judged.verdict == "supported"
        for cite in claim.citations
        if cite in tools.made
    ]
    written: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for cite in dict.fromkeys(supported):
        computed = tools.made[cite]
        planned = computed.plan
        ids = [_fact_id(conn, planned, f) for f in planned.fetches]
        if None in ids:
            skipped.append(
                {"metric": planned.metric.name, "reason": "an input is no longer verified"}
            )
            continue
        start, end = span(planned.year, planned.period)
        own = [
            json.loads(tools.evidence[c].text)
            for f, (_, c) in zip(planned.fetches, computed.inputs, strict=True)
            if (f.year, f.period) == (planned.year, planned.period)
        ]
        instant = bool(own) and all("as_at" in cast(dict[str, object], f["period"]) for f in own)
        periods = _periods(planned.year, planned.period, planned.previous, planned.snapshot)
        row = DerivedRow(
            ticker=planned.company,
            basis=planned.basis,
            concept=planned.metric.name,
            label=planned.metric.label
            if planned.previous is None
            else f"{planned.metric.label}, {periods}",
            period_start=None if instant else start.isoformat(),
            period_end=end.isoformat(),
            segment=None,
            value=computed.value,
            unit=computed.unit,
            formula=json.dumps(
                {
                    "metric": planned.metric.name,
                    "scope": planned.metric.scope,
                    "formula": planned.metric.formula,
                    "inputs": dict(computed.inputs),
                    "periods": periods,
                },
                sort_keys=True,
            ),
        )
        with conn:
            checked = planned.metric.verified_by
            trusted = planned.metric.tier == "derived-concept" and checked is not None
            fact_id = write_derived(
                conn,
                row,
                [i for i in ids if i is not None],
                writer,
                concept_check(planned.metric.name, planned.metric.scope, checked)
                if trusted and checked
                else None,
            )
        written.append(
            {
                "fact_id": fact_id,
                "cite_as": cite,
                "metric": planned.metric.name,
                "label": planned.metric.label,
                "value": computed.value,
                "unit": computed.unit,
                "tier": "derived" if trusted else "derived-ai",
                "formula": planned.metric.formula,
                "periods": periods,
                "inputs": [c for _, c in computed.inputs],
            }
        )
    return written, skipped


def _fact_id(conn: sqlite3.Connection, planned: Plan, fetch: Fetch) -> int | None:
    """The checked fact a fetch names, as `fundamentals` finds it, or None."""
    start, end = span(fetch.year, fetch.period)
    row = conn.execute(
        f"""SELECT id FROM facts WHERE ticker = ? AND basis = ? AND concept = ?
              AND segment IS NULL AND tier IN ({", ".join("?" for _ in CHECKED_TIERS)})
              AND ((period_kind = 'duration' AND period_start = ? AND period_end = ?)
                   OR (period_kind = 'instant' AND period_end = ?))
            ORDER BY id LIMIT 1""",
        (
            planned.company,
            planned.basis,
            fetch.concept,
            *CHECKED_TIERS,
            start.isoformat(),
            end.isoformat(),
            end.isoformat(),
        ),
    ).fetchone()
    return None if row is None else int(row[0])
