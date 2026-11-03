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
    OFFSETS,
    Binding,
    DerivedConcept,
    Fetch,
    Offset,
    Step,
    available,
    bindings_json,
    bindings_refusal,
    evaluate,
    load_derived_concepts,
    matched,
    name_refusal,
    plan,
    ranked,
)
from aigent.facts.formulas import FORMULAS
from aigent.facts.lookup import BASES, PERIODS, Period, span
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
        "Step 1 of computing a figure that needs arithmetic (a growth rate, a ratio, a margin): "
        "find the metric for it and the exact facts it needs for one company, year, period and "
        "basis. Returns a plan_id, the metric's library formula, and a fetch list: call "
        "fundamentals once per entry with exactly the arguments given, then compute_metric with "
        "the plan_id and each fact's cite_as. A metric is found only by its exact name or when "
        "its name and label hold every word asked; otherwise the answer lists the closest of the "
        "company's scope and the formula library, and you plan one by name or define it with "
        "define_metric. Example: plan_metric(what='operating profit growth', company='ICICIBANK', "
        "year='2025', period='year', basis='standalone')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "what": {"type": "string", "description": "The figure, in words or a metric name."},
            "company": {"type": "string", "enum": list(FACTS_TICKERS)},
            "year": {"type": "string", "description": "The financial year it ends in: 2025."},
            "period": {"type": "string", "enum": list(PERIODS)},
            "basis": {"type": "string", "enum": list(BASES)},
        },
        "required": ["what", "company", "year", "period", "basis"],
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
        "an offset: 't' the period asked, 't-1y' a year before, 't-1q' the quarter before. It is "
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
                        "offset": {"type": "string", "enum": list(OFFSETS)},
                    },
                    "required": ["variable", "concept", "offset"],
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
        what, company = str(arguments.get("what", "")).strip(), str(arguments.get("company", ""))
        year, period = str(arguments.get("year", "")), str(arguments.get("period", ""))
        basis = str(arguments.get("basis", ""))
        if not what or company not in FACTS_TICKERS or not year.isdigit():
            raise ValueError("give what, a company of the database and the year as digits")
        if period not in PERIODS or basis not in BASES:
            raise ValueError(f"period is one of {', '.join(PERIODS)}; basis one of {BASES}")
        scope = COMPANY_SCOPES[company]
        held = load_derived_concepts(self.conn, scope)
        metric = matched(what, held)
        if metric is None:
            return {
                "metric": None,
                "scope": scope,
                "note": (
                    "No metric in this scope matches every word. If one of the closest is meant, "
                    "plan it by its exact name; else define one with define_metric."
                ),
                "closest": [{"name": m.name, "label": m.label} for m in ranked(what, held)[:5]],
                "library": _LIBRARY,
            }
        step = plan(metric, int(year), cast(Period, period), {m.name: m for m in held})
        plan_id = f"P{len(self.plans) + 1}"
        self.plans[plan_id] = Plan(metric, company, int(year), cast(Period, period), basis, step)
        return {
            "plan_id": plan_id,
            "metric": metric.name,
            "scope": scope,
            "label": metric.label,
            "tier": metric.tier,
            "formula": f"{metric.formula}: {FORMULAS[metric.formula].description}",
            "fetch": [
                {
                    "variable": f.variable,
                    "fundamentals": {
                        "company": company,
                        "year": str(f.year),
                        "what": f.concept,
                        "period": f.period,
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
            "year": planned.year,
            "period": planned.period,
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
            bindings[variable] = Binding(
                str(row.get("concept", "")), cast(Offset, row.get("offset"))
            )
        if why := bindings_refusal(formula, bindings, available(self.conn, scope), name):
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
        row = DerivedRow(
            ticker=planned.company,
            basis=planned.basis,
            concept=planned.metric.name,
            label=planned.metric.label,
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
                "period": f"{planned.period} {planned.year}",
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
