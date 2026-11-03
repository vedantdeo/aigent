"""Derived concepts, which the agent calls metrics: a named figure defined as a library formula
over concepts, each input at a period relative to the one asked for, with no company and no period
in the definition. A plan turns one into the exact facts to fetch for a company, year, period and
basis."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

from aigent.facts.formulas import FORMULAS
from aigent.facts.lookup import Period
from aigent.facts.store import Writer, write_derived_concept

SEEDED = Path(__file__).resolve().parents[3] / "evals" / "reference" / "derived-concepts.jsonl"

Offset = Literal["t", "t-1y", "t-1q"]  # the period asked; a year before; the quarter before
OFFSETS: tuple[Offset, ...] = get_args(Offset)
_QUARTERS: tuple[Period, ...] = ("Q1", "Q2", "Q3", "Q4")
_CAMEL = re.compile(r"[A-Z][A-Z0-9]*(?![a-z])|[A-Z]?[a-z0-9]+")
_NAME = re.compile(r"[A-Z][A-Za-z0-9]*")
_PERIOD_IN_NAME = re.compile(r"FY\d|20\d\d|Q[1-4](?!\d)|H1|9M")


@dataclass(frozen=True)
class Binding:
    """One input of a metric's formula: a concept, at a period relative to the one asked."""

    concept: str
    offset: Offset


@dataclass(frozen=True)
class DerivedConcept:
    name: str
    label: str
    formula: str
    bindings: Mapping[str, Binding]
    tier: str


@dataclass(frozen=True)
class Fetch:
    """One input to fetch: which variable it fills, and the fact that fills it."""

    variable: str
    concept: str
    year: int
    period: Period

    def cite_as(self, company: str, basis: str) -> str:
        """The cite `fundamentals` gives this fact."""
        return f"FACT:{company}:{self.year}:{basis}:{self.period}:{self.concept}"


def shift(year: int, period: Period, offset: Offset) -> tuple[int, Period]:
    """The financial year and period `offset` names, from the one asked. ValueError for a quarter
    before a period that is not a quarter."""
    if offset == "t":
        return year, period
    if offset == "t-1y":
        return year - 1, period
    if period not in _QUARTERS:
        raise ValueError(f"a quarter before {period} is not a quarter; ask for Q1 to Q4")
    at = _QUARTERS.index(period)
    return (year - 1, "Q4") if at == 0 else (year, _QUARTERS[at - 1])


def plan(metric: DerivedConcept, year: int, period: Period) -> list[Fetch]:
    """Each input of `metric` as the fact to fetch for `year` and `period`, in formula order."""
    out: list[Fetch] = []
    for variable in FORMULAS[metric.formula].variables:
        bound = metric.bindings[variable]
        at_year, at_period = shift(year, period, bound.offset)
        out.append(Fetch(variable, bound.concept, at_year, at_period))
    return out


def words(text: str) -> set[str]:
    """Lower-case words of a label or a CamelCase name."""
    found = _CAMEL.findall(text) + re.findall(r"[A-Za-z0-9]+", text)
    return {w.lower() for w in found}


def ranked(what: str, held: Sequence[DerivedConcept]) -> list[DerivedConcept]:
    """The derived concepts sharing a word with `what`, most shared first, then by name; an exact
    name first of all."""
    asked = words(what)
    scored = [(m, len(asked & (words(m.name) | words(m.label)))) for m in held]
    kept = [(m, shared) for m, shared in scored if m.name == what or shared > 0]
    return [m for m, shared in sorted(kept, key=lambda k: (k[0].name != what, -k[1], k[0].name))]


def name_refusal(name: str) -> str | None:
    """Why `name` cannot name a metric, or None: CamelCase, and no year or period in it."""
    if not _NAME.fullmatch(name):
        return f"{name!r} is not a CamelCase name such as NetInterestIncome"
    if _PERIOD_IN_NAME.search(name):
        return f"{name!r} names a period; a metric holds for every period"
    return None


def bindings_refusal(
    formula: str, bindings: Mapping[str, Binding], concepts: set[str]
) -> str | None:
    """Why `bindings` cannot fill `formula`, or None."""
    if formula not in FORMULAS:
        return f"no formula {formula!r}; one of {', '.join(FORMULAS)}"
    wanted = set(FORMULAS[formula].variables)
    if set(bindings) != wanted:
        return f"{formula} takes {', '.join(sorted(wanted))}, not {', '.join(sorted(bindings))}"
    unknown = sorted(b.concept for b in bindings.values() if b.concept not in concepts)
    if unknown:
        return f"no concept {', '.join(unknown)} in the database"
    odd = sorted(b.offset for b in bindings.values() if b.offset not in OFFSETS)
    if odd:
        return f"offset {', '.join(odd)} is not one of {', '.join(OFFSETS)}"
    return None


def as_bindings(text: str) -> dict[str, Binding]:
    """A metric's stored bindings, `{variable: {concept, offset}}`, as Bindings."""
    raw = cast(dict[str, dict[str, str]], json.loads(text))
    return {v: Binding(b["concept"], cast(Offset, b["offset"])) for v, b in raw.items()}


def bindings_json(bindings: Mapping[str, Binding]) -> str:
    """Bindings as stored: variables sorted, each `{concept, offset}`."""
    return json.dumps(
        {v: {"concept": b.concept, "offset": b.offset} for v, b in sorted(bindings.items())}
    )


def load_derived_concepts(conn: sqlite3.Connection) -> list[DerivedConcept]:
    """Every metric held, by name."""
    rows = conn.execute(
        "SELECT name, label, formula, bindings, tier FROM derived_concepts ORDER BY name"
    )
    return [DerivedConcept(n, label, f, as_bindings(b), tier) for n, label, f, b, tier in rows]


def seed(conn: sqlite3.Connection, writer: Writer, path: Path = SEEDED) -> list[str]:
    """Writes each derived concept in `path` not yet held, as `derived-concept-ai`; returns why
    any could not be."""
    concepts = {r[0] for r in conn.execute("SELECT concept FROM concepts")}
    refused: list[str] = []
    with conn:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = cast(dict[str, object], json.loads(line))
            name, formula = str(row["name"]), str(row["formula"])
            bindings = as_bindings(json.dumps(row["bindings"]))
            why = name_refusal(name) or bindings_refusal(formula, bindings, concepts)
            if why:
                refused.append(f"{name}: {why}")
                continue
            write_derived_concept(
                conn, name, str(row["label"]), formula, bindings_json(bindings), None, writer
            )
    return refused
