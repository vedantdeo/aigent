"""Derived concepts, which the agent calls metrics: a named figure defined as a library formula
over concepts, each input at the current period or the previous one, both named when it is
planned, with no company and no period in the definition. Each holds within a scope, the XBRL
taxonomy its companies file under, since banks name their figures differently. An input may itself
be a derived concept of the same scope; a plan turns one into the exact facts to fetch for a
company, periods and basis."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, get_args

from aigent.facts.cfg_facts import COMPANY_SCOPES
from aigent.facts.formulas import FORMULAS
from aigent.facts.lookup import Period, span
from aigent.facts.store import (
    XBRL_TIERS,
    Writer,
    covers,
    normal_scope,
    scope_refusal,
    scopes_of,
    write_derived_concept,
)

SEEDED = Path(__file__).resolve().parents[3] / "evals" / "reference" / "derived-concepts.jsonl"

At = Literal["current", "previous"]
ATS: tuple[At, ...] = get_args(At)
When = tuple[int, Period]  # a financial year and a period of it
_QUARTERS: tuple[Period, ...] = ("Q1", "Q2", "Q3", "Q4")
_CAMEL = re.compile(r"[A-Z][A-Z0-9]*(?![a-z])|[A-Z]?[a-z0-9]+")
_NAME = re.compile(r"[A-Z][A-Za-z0-9]*")
_PERIOD_IN_NAME = re.compile(r"FY\d|20\d\d|Q[1-4](?!\d)|H1|9M")
_FILLER = {"a", "an", "and", "as", "for", "in", "of", "over", "per", "the", "to", "what", "was"}


@dataclass(frozen=True)
class Binding:
    """One input of a formula: a concept, filed or derived, at the current or the previous
    period."""

    concept: str
    at: At


@dataclass(frozen=True)
class DerivedConcept:
    name: str
    scope: str
    label: str
    formula: str
    bindings: Mapping[str, Binding]
    tier: str
    verified_by: str | None = None


@dataclass(frozen=True)
class Fetch:
    """One fact to fetch: the input it fills, by path (`denominator.first`), and the fact."""

    variable: str
    concept: str
    year: int
    period: Period

    def cite_as(self, company: str, basis: str) -> str:
        """The cite `fundamentals` gives this fact."""
        return f"FACT:{company}:{self.year}:{basis}:{self.period}:{self.concept}"


@dataclass(frozen=True)
class Step:
    """A plan: a derived concept at a period, its inputs a fetched fact or a step of their own."""

    concept: DerivedConcept
    year: int
    period: Period
    inputs: Mapping[str, Fetch | Step] = field(default_factory=dict[str, "Fetch | Step"])
    previous: When | None = None

    def fetches(self) -> list[Fetch]:
        """Every fact the plan reads, in formula order, depth first."""
        out: list[Fetch] = []
        for value in self.inputs.values():
            out.extend(value.fetches() if isinstance(value, Step) else [value])
        return out


def compares(concept: DerivedConcept) -> bool:
    """Whether `concept` reads the previous period as well as the current one."""
    return any(b.at == "previous" for b in concept.bindings.values())


Kind = Literal["flow", "snapshot", "mixed"]


def leaves(concept: DerivedConcept, held: Mapping[str, DerivedConcept]) -> set[str]:
    """The filed concepts `concept` reads, through any derived ones."""
    found: set[str] = set()
    for bound in concept.bindings.values():
        inner = held.get(bound.concept)
        found |= leaves(inner, held) if inner is not None else {bound.concept}
    return found


def kind(
    conn: sqlite3.Connection, concept: DerivedConcept, held: Mapping[str, DerivedConcept]
) -> Kind:
    """`snapshot` when every concept it reads is a balance as at a date, `flow` when none is,
    else `mixed`; a concept is what most of its facts are."""
    read = sorted(leaves(concept, held))
    rows = conn.execute(
        f"""SELECT concept, SUM(period_kind = 'instant') > SUM(period_kind = 'duration')
            FROM facts WHERE concept IN ({", ".join("?" for _ in read)}) GROUP BY concept""",
        read,
    ).fetchall()
    instant = {str(c) for c, at in rows if at}
    if instant == set(read):
        return "snapshot"
    return "mixed" if instant else "flow"


def interval_refusal(current: When, previous: When) -> str | None:
    """Why `previous` cannot be compared with `current`, or None: it ends before `current` begins
    and is as long, a quarter against any quarter."""
    (year, period), (was_year, was_period) = current, previous
    alike = period == was_period or (period in _QUARTERS and was_period in _QUARTERS)
    if not alike:
        return f"{was_period} {was_year} is not as long as {period} {year}; compare like with like"
    if span(was_year, was_period)[1] >= span(year, period)[0]:
        return f"{was_period} {was_year} does not end before {period} {year} begins"
    return None


def plan(
    concept: DerivedConcept,
    year: int,
    period: Period,
    held: Mapping[str, DerivedConcept] | None = None,
    previous: When | None = None,
    path: str = "",
) -> Step:
    """`concept` at `year` and `period`, each input a fact to fetch or, when it names a derived
    concept in `held`, that concept's own plan at the input's period. ValueError when `previous`
    is missing for a concept that compares two periods, given for one that does not, or no
    earlier period like the current one."""
    if not path:
        if compares(concept) and previous is None:
            raise ValueError(f"{concept.name} compares two periods; give the previous period too")
        if not compares(concept) and previous is not None:
            raise ValueError(f"{concept.name} is at one period; give no previous period")
        if previous is not None and (why := interval_refusal((year, period), previous)):
            raise ValueError(why)
    inputs: dict[str, Fetch | Step] = {}
    for variable in FORMULAS[concept.formula].variables:
        bound = concept.bindings[variable]
        at_year, at_period = (year, period) if bound.at == "current" else cast(When, previous)
        where = f"{path}{variable}"
        inner = (held or {}).get(bound.concept)
        if inner is not None and compares(inner):
            raise ValueError(f"{inner.name} compares two periods; it cannot be an input")
        inputs[variable] = (
            plan(inner, at_year, at_period, held, None, f"{where}.")
            if inner is not None
            else Fetch(where, bound.concept, at_year, at_period)
        )
    return Step(concept, year, period, inputs, previous if path == "" else None)


def evaluate(step: Step, facts: Mapping[str, tuple[float, str]]) -> tuple[float, str]:
    """The plan's value and unit from each fetched fact's `(value, unit)` by path. ValueError for
    a zero base or a formula needing one unit given two."""
    values: dict[str, float] = {}
    units: set[str] = set()
    for variable, given in step.inputs.items():
        value, unit = evaluate(given, facts) if isinstance(given, Step) else facts[given.variable]
        values[variable] = value
        units.add(unit)
    formula = FORMULAS[step.concept.formula]
    if formula.unit == "same" and len(units) > 1:
        raise ValueError(f"{formula.name} needs one unit, not {', '.join(sorted(units))}")
    return formula.compute(**values), "pure" if formula.unit == "pure" else units.pop()


def words(text: str) -> set[str]:
    """Lower-case words of a label or a CamelCase name."""
    found = _CAMEL.findall(text) + re.findall(r"[A-Za-z0-9]+", text)
    return {w.lower() for w in found}


def ranked(what: str, held: Sequence[DerivedConcept]) -> list[DerivedConcept]:
    """The derived concepts sharing a word with `what`, most shared first, then by name; an exact
    name first of all."""
    asked = words(what) - _FILLER
    scored = [(m, len(asked & (words(m.name) | words(m.label)))) for m in held]
    kept = [(m, shared) for m, shared in scored if m.name == what or shared > 0]
    return [m for m, shared in sorted(kept, key=lambda k: (k[0].name != what, -k[1], k[0].name))]


def matched(what: str, held: Sequence[DerivedConcept]) -> DerivedConcept | None:
    """The derived concept `what` names exactly, or the best whose name and label hold every word
    of `what` but filler; None when none does, so a near miss is never planned."""
    for concept in held:
        if concept.name == what:
            return concept
    asked = words(what) - _FILLER
    for concept in ranked(what, held):
        if asked and asked <= words(concept.name) | words(concept.label):
            return concept
    return None


def name_refusal(name: str) -> str | None:
    """Why `name` cannot name a metric, or None: CamelCase, and no year or period in it."""
    if not _NAME.fullmatch(name):
        return f"{name!r} is not a CamelCase name such as NetInterestIncome"
    if _PERIOD_IN_NAME.search(name):
        return f"{name!r} names a period; a metric holds for every period"
    return None


def bindings_refusal(
    formula: str,
    bindings: Mapping[str, Binding],
    concepts: set[str],
    name: str = "",
    comparing: set[str] | None = None,
) -> str | None:
    """Why `bindings` cannot fill `formula` with the scope's `concepts`, or None. A concept naming
    the one being defined is refused, so no definition can loop, and so is one of `comparing`,
    which compares two periods itself."""
    if formula not in FORMULAS:
        return f"no formula {formula!r}; one of {', '.join(FORMULAS)}"
    wanted = set(FORMULAS[formula].variables)
    if set(bindings) != wanted:
        return f"{formula} takes {', '.join(sorted(wanted))}, not {', '.join(sorted(bindings))}"
    if any(b.concept == name for b in bindings.values()):
        return f"{name} cannot be an input to itself"
    unknown = sorted(b.concept for b in bindings.values() if b.concept not in concepts)
    if unknown:
        return f"no concept {', '.join(unknown)} filed or derived in this scope"
    odd = sorted(b.at for b in bindings.values() if b.at not in ATS)
    if odd:
        return f"at {', '.join(odd)} is not one of {', '.join(ATS)}"
    twice = sorted(b.concept for b in bindings.values() if b.concept in (comparing or set()))
    if twice:
        return f"{', '.join(twice)} compares two periods itself; it cannot be an input"
    return None


def available(conn: sqlite3.Connection, scope: str) -> set[str]:
    """The concepts an input may name in `scope` (a spec, `*` or a list): those filed or derived
    in every scope it names."""
    found: set[str] | None = None
    held = conn.execute("SELECT name, scope FROM derived_concepts").fetchall()
    for one in scopes_of(scope):
        tickers = [t for t, s in COMPANY_SCOPES.items() if s == one]
        filed = conn.execute(
            f"""SELECT DISTINCT concept FROM facts
                WHERE ticker IN ({", ".join("?" for _ in tickers)})
                  AND tier IN ({", ".join("?" for _ in XBRL_TIERS)})""",
            (*tickers, *XBRL_TIERS),
        )
        here = {r[0] for r in filed} | {n for n, spec in held if covers(spec, one)}
        found = here if found is None else found & here
    return found or set()


def comparing(conn: sqlite3.Connection, scope: str) -> set[str]:
    """The derived concepts held in any scope `scope` names that compare two periods."""
    return {
        m.name for one in scopes_of(scope) for m in load_derived_concepts(conn, one) if compares(m)
    }


def as_bindings(text: str) -> dict[str, Binding]:
    """A metric's stored bindings, `{variable: {concept, at}}`, as Bindings."""
    raw = cast(dict[str, dict[str, str]], json.loads(text))
    return {v: Binding(b["concept"], cast(At, b["at"])) for v, b in raw.items()}


def bindings_json(bindings: Mapping[str, Binding]) -> str:
    """Bindings as stored: variables sorted, each `{concept, at}`."""
    return json.dumps({v: {"at": b.at, "concept": b.concept} for v, b in sorted(bindings.items())})


def load_derived_concepts(
    conn: sqlite3.Connection, scope: str | None = None
) -> list[DerivedConcept]:
    """Every derived concept held, or those whose scope covers `scope`, by name."""
    rows = conn.execute(
        """SELECT name, scope, label, formula, bindings, tier, verified_by FROM derived_concepts
           ORDER BY name, scope"""
    )
    return [
        DerivedConcept(n, sc, label, f, as_bindings(b), tier, by)
        for n, sc, label, f, b, tier, by in rows
        if scope is None or covers(sc, scope)
    ]


def seed(conn: sqlite3.Connection, writer: Writer, path: Path = SEEDED) -> list[str]:
    """Writes each derived concept in `path` not yet held, as `derived-concept-ai`, in passes until
    one writes nothing, so a concept may build on one later in the file; returns why any could
    not be."""
    pending = [cast(dict[str, object], json.loads(line)) for line in path.read_text().splitlines()]
    while True:
        refused: list[str] = []
        waiting: list[dict[str, object]] = []
        with conn:
            for row in pending:
                name, scope, formula = str(row["name"]), str(row["scope"]), str(row["formula"])
                bindings = as_bindings(json.dumps(row["bindings"]))
                why = (
                    scope_refusal(scope)
                    or name_refusal(name)
                    or bindings_refusal(
                        formula, bindings, available(conn, scope), name, comparing(conn, scope)
                    )
                )
                if why:
                    refused.append(f"{name} [{scope}]: {why}")
                    waiting.append(row)
                    continue
                written = write_derived_concept(
                    conn,
                    name,
                    scope,
                    str(row["label"]),
                    formula,
                    bindings_json(bindings),
                    None,
                    writer,
                )
                held = conn.execute(
                    "SELECT 1 FROM derived_concepts WHERE name = ? AND scope = ?",
                    (name, normal_scope(scope)),
                ).fetchone()
                if not written and held is None:
                    refused.append(f"{name} [{scope}]: overlaps a scope it is already held in")
        if len(waiting) == len(pending):
            return refused
        pending = waiting
