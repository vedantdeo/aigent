"""The coverage eval: of each labelled figure, is it held in the facts database, found by the lookup
for a question's words, and exact. Free; `python -m aigent.facts coverage` writes the report."""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from aigent.facts.cfg_facts import (
    COVERAGE_DEFINITIONS,
    COVERAGE_FOUND_AT,
    COVERAGE_MAX_ROUNDING,
    COVERAGE_RESULTS_METRICS,
    COVERAGE_TEACHER_UNITS,
    FACTS_TICKERS,
    FACTS_YEARS,
    FUNDAMENTALS_CONCEPTS,
)
from aigent.facts.check import tolerance
from aigent.facts.lookup import Lookup, NotFound, span
from aigent.facts.reference import Fundamental
from aigent.facts.store import CHECKED_TIERS

REPO = Path(__file__).resolve().parents[3]
TABLE_QUESTIONS = REPO / "evals" / "datasets" / "retrieval-tables.jsonl"
TABLE_FIGURES = REPO / "evals" / "reference" / "table-figures.json"
TEACHER_LABELS = REPO / "corpus" / "training" / "figure-answers.jsonl"
REPORTS = REPO / "evals" / "reports"

Phrasing = Literal["asked", "word"]
PHRASINGS: tuple[Phrasing, ...] = ("asked", "word")
Match = Literal["concept", "value"]
SOURCES = ("fundamentals", "tables", "teacher-results", "teacher-other")


@dataclass(frozen=True)
class Figure:
    """One labelled figure: whose, which year, its value in full units, and the words to ask by.
    Matched by `concept` when `match` is "concept" (None: not filed), else by value."""

    id: str
    source: str
    ticker: str
    year: int
    value: float
    tolerance: float
    asks: Mapping[Phrasing, str]
    match: Match
    concept: str | None = None
    units: tuple[str, ...] = ()


@dataclass(frozen=True)
class Left:
    """A label not scored, and why."""

    id: str
    source: str
    reason: str


@dataclass(frozen=True)
class Holding:
    """A fact in the database that holds a figure."""

    concept: str
    basis: str
    segment: str | None
    value: float
    decimals: int | None


@dataclass(frozen=True)
class Scored:
    """One figure asked one way: held, the rank it was found at, and exact (None: not measured)."""

    id: str
    source: str
    phrasing: Phrasing
    held: bool
    rank: int | None
    exact: bool | None
    detail: str


# Loading the three sources.


def fundamentals_figures(table: Iterable[Fundamental]) -> list[Figure]:
    """The frozen fundamentals, each matched by its XBRL concept and asked by its definition."""
    return [
        Figure(
            id=item.key,
            source="fundamentals",
            ticker=item.ticker,
            year=item.year,
            value=item.value,
            tolerance=0.0,
            asks={"asked": COVERAGE_DEFINITIONS[item.metric], "word": item.metric},
            match="concept",
            concept=FUNDAMENTALS_CONCEPTS.get(item.metric),
        )
        for item in table
    ]


def _decimals(number: str) -> int:
    return len(number.split(".")[1]) if "." in number else 0


def _outside(ticker: str, year: int) -> str | None:
    if ticker not in FACTS_TICKERS:
        return "not a company the database holds"
    if year not in FACTS_YEARS:
        return "not a year the database holds"
    return None


def table_figures(
    questions: Path = TABLE_QUESTIONS, figures: Path = TABLE_FIGURES
) -> tuple[list[Figure], list[Left]]:
    """Each table question's hand-written figure, matched by value and asked as written."""
    asked = {
        str(row["id"]): str(row["input"]["question"])
        for line in questions.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
        for row in [json.loads(line)]
    }
    entries = cast(
        dict[str, dict[str, object]],
        json.loads(figures.read_text(encoding="utf-8"))["figures"],
    )
    scored: list[Figure] = []
    left: list[Left] = []
    for id_, entry in entries.items():
        if id_ not in asked:
            raise ValueError(f"{figures.name}: {id_} is not a table question")
        ticker, year = str(entry["ticker"]), int(cast(int, entry["year"]))
        reason = cast(str | None, entry["excluded"]) or _outside(ticker, year)
        if reason is not None:
            left.append(Left(id_, "tables", reason))
            continue
        printed, scale = str(entry["printed"]), float(cast(float, entry["scale"]))
        scored.append(
            Figure(
                id=id_,
                source="tables",
                ticker=ticker,
                year=year,
                value=float(printed) * scale,
                tolerance=0.5 * 10.0 ** -_decimals(printed) * scale,
                asks={"asked": asked[id_], "word": str(entry["metric"])},
                match="value",
                units=(str(entry["unit"]),),
            )
        )
    return scored, left


def _factors(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return [*_factors(node.left), *_factors(node.right)]
    return [node]


def _evaluate(node: ast.expr) -> float:
    """A multiplier: integers, `**` and `*` only."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return float(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.Mult)):
        left, right = _evaluate(node.left), _evaluate(node.right)
        return left**right if isinstance(node.op, ast.Pow) else left * right
    raise ValueError("not a multiplier")


def tolerance_of(call: str | None, value: float) -> float | None:
    """Half a unit of the label's last written digit, in full units: from its `call` (a number,
    times any multipliers), else from the value; None when the call is not that shape."""
    if call is None:
        return 0.5 * 10.0 ** -_decimals(repr(value))
    text = call.replace(",", "")
    try:
        tree = ast.parse(text, mode="eval").body
        first, *rest = _factors(tree)
        multiplier = 1.0
        for factor in rest:
            multiplier *= _evaluate(factor)
    except (SyntaxError, ValueError):
        return None
    written = ast.get_source_segment(text, first) or ""
    numbers = re.findall(r"\d+(?:\.\d+)?", written)
    if re.search(r"[^\d.\s()+-]", written):
        return None
    return 0.5 * 10.0 ** -max(_decimals(n) for n in numbers) * multiplier


def _ticker(chunk_id: str) -> str:
    return chunk_id.split("#")[0].rsplit("-", 1)[0]


def teacher_figures(path: Path = TEACHER_LABELS) -> tuple[list[Figure], list[Left]]:
    """The v3 teacher's labelled amounts with a named metric, matched by value; results-format
    metrics and the rest are two sources. A figure repeated with the same question counts once."""
    scored: list[Figure] = []
    left: list[Left] = []
    seen: set[tuple[str, int, str, float, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = cast(dict[str, object], json.loads(line))
        chunk_id, question = str(row["chunk_id"]), str(row["question"])
        figures = cast(list[dict[str, object]], row.get("figures") or [])
        for i, figure in enumerate(figures):
            id_ = f"{chunk_id}/{i}"
            metric = str(figure.get("metric") or "other")
            source = "teacher-results" if metric in COVERAGE_RESULTS_METRICS else "teacher-other"
            ticker = _ticker(chunk_id)
            year, value = figure.get("year"), figure.get("value")
            if metric == "other":
                left.append(Left(id_, "teacher", "no named metric"))
                continue
            if not isinstance(year, int) or not isinstance(value, (int, float)):
                left.append(Left(id_, source, "no year or no value"))
                continue
            outside = _outside(ticker, year)
            if outside is not None:
                left.append(Left(id_, source, outside))
                continue
            if figure.get("unit") != "INR" or figure.get("description") == "change":
                left.append(Left(id_, source, "not an amount in rupees"))
                continue
            if value == 0:
                left.append(Left(id_, source, "zero, which any nil fact would match"))
                continue
            tol = tolerance_of(cast(str | None, figure.get("call")), float(value))
            if tol is None:
                left.append(Left(id_, source, "its precision cannot be read from its call"))
                continue
            key = (ticker, year, metric, float(value), question)
            if key in seen:
                left.append(Left(id_, source, "repeats a figure already counted"))
                continue
            seen.add(key)
            scored.append(
                Figure(
                    id=id_,
                    source=source,
                    ticker=ticker,
                    year=year,
                    value=float(value),
                    tolerance=tol,
                    asks={"asked": question, "word": metric.replace("_", " ")},
                    match="value",
                    units=COVERAGE_TEACHER_UNITS.get(metric, ("INR",)),
                )
            )
    return scored, left


# Scoring.


def _equal(found: float, figure: Figure, decimals: int | None) -> bool:
    """Equal within both roundings; by value, the sign is ignored, which presentations differ on."""
    allowed = figure.tolerance + tolerance([decimals])
    if figure.match == "value":
        return abs(abs(found) - abs(figure.value)) <= allowed
    return abs(found - figure.value) <= allowed


def _telling(value: float, figure: Figure, decimals: int | None) -> bool:
    """Equal by value, and filed precisely enough for that to say something."""
    rounding = tolerance([decimals])
    return rounding <= COVERAGE_MAX_ROUNDING * abs(figure.value) and _equal(value, figure, decimals)


def holdings(conn: sqlite3.Connection, figure: Figure) -> list[Holding]:
    """The answerable facts for the figure's company and year that hold it, consolidated and
    unsegmented first, each concept once from its earliest filing."""
    start, end = span(figure.year, "year")
    checked = ", ".join("?" for _ in CHECKED_TIERS)
    rows = conn.execute(
        f"""SELECT f.concept, f.basis, f.segment, f.value, f.decimals, f.unit
            FROM facts f LEFT JOIN filings fl ON fl.id = f.filing_id
            WHERE f.ticker = ? AND (f.tier = 'xbrl' OR f.tier IN ({checked}))
              AND ((f.period_kind = 'duration' AND f.period_start = ? AND f.period_end = ?)
                   OR (f.period_kind = 'instant' AND f.period_end = ?))
            ORDER BY f.basis != 'consolidated', f.segment IS NOT NULL, f.segment, f.concept,
                     fl.period_end""",
        (figure.ticker, *CHECKED_TIERS, start.isoformat(), end.isoformat(), end.isoformat()),
    ).fetchall()
    held: dict[tuple[str, str, str | None], Holding] = {}
    for concept, basis, segment, value, decimals, unit in rows:
        key = (concept, basis, segment)
        if key in held:
            continue
        if figure.match == "concept":
            if concept != figure.concept or basis != "consolidated" or segment is not None:
                continue
        elif unit not in figure.units or not _telling(float(value), figure, decimals):
            continue
        held[key] = Holding(concept, basis, segment, float(value), decimals)
    return list(held.values())


def score(conn: sqlite3.Connection, lookup: Lookup, figure: Figure, phrasing: Phrasing) -> Scored:
    """The figure asked one way, with the basis and segment of the fact that holds it."""
    held = holdings(conn, figure)

    def scored(rank: int | None, exact: bool | None, detail: str) -> Scored:
        return Scored(figure.id, figure.source, phrasing, bool(held), rank, exact, detail)

    if not held:
        return scored(None, None if figure.match == "value" else False, "not held")
    best = held[0]
    try:
        found = lookup(
            figure.ticker,
            figure.year,
            figure.asks[phrasing],
            basis=cast(Literal["consolidated", "standalone"], best.basis),
            period="year",
            segment=best.segment,
        )
    except NotFound:
        found = []
    holding = {(h.concept, h.basis, h.segment): h for h in held}
    rank = next(
        (i for i, f in enumerate(found, 1) if (f.concept, f.basis, f.segment) in holding), None
    )
    where = f"{best.concept}{'' if best.segment is None else ' / ' + best.segment} ({best.basis})"
    if rank is None:
        returned = "; ".join(f.label for f in found[:3]) or "nothing"
        return scored(
            None, None if figure.match == "value" else False, f"held as {where}; got {returned}"
        )
    if figure.match == "value":
        return scored(rank, None, f"rank {rank}: {where}")
    hit = found[rank - 1]
    exact = _equal(hit.value, figure, holding[(hit.concept, hit.basis, hit.segment)].decimals)
    detail = (
        f"rank {rank}: {where}"
        if exact
        else f"rank {rank}: {hit.value:,.2f} for {figure.value:,.2f}"
    )
    return scored(rank, exact, detail)


def evaluate(conn: sqlite3.Connection, lookup: Lookup, figures: Sequence[Figure]) -> list[Scored]:
    """Every figure asked every way; the lookups it logged deleted after, whatever happens."""
    try:
        return [
            score(conn, lookup, figure, phrasing) for figure in figures for phrasing in PHRASINGS
        ]
    finally:
        with conn:
            conn.execute("DELETE FROM lookups WHERE run_id = ?", (lookup.run_id,))


# The report.


@dataclass(frozen=True)
class Line:
    """One source asked one way: how many figures, held, found at each rank, and exact."""

    source: str
    phrasing: Phrasing
    figures: int
    held: int
    found: Mapping[int, int]
    exact: int | None


def summarise(scored: Iterable[Scored]) -> list[Line]:
    """One line per source and phrasing, in `SOURCES` then `PHRASINGS` order."""
    groups: dict[tuple[str, Phrasing], list[Scored]] = {}
    for row in scored:
        groups.setdefault((row.source, row.phrasing), []).append(row)
    lines: list[Line] = []
    for source in SOURCES:
        for phrasing in PHRASINGS:
            rows = groups.get((source, phrasing), [])
            if not rows:
                continue
            measured = [r.exact for r in rows if r.held and r.exact is not None]
            lines.append(
                Line(
                    source=source,
                    phrasing=phrasing,
                    figures=len(rows),
                    held=sum(r.held for r in rows),
                    found={
                        k: sum(r.rank is not None and r.rank <= k for r in rows)
                        for k in COVERAGE_FOUND_AT
                    },
                    exact=sum(measured) if any(r.exact is not None for r in rows) else None,
                )
            )
    return lines


def _rate(part: int, whole: int) -> str:
    return "—" if whole == 0 else f"{part}/{whole} ({100 * part / whole:.0f}%)"


def _table(lines: Sequence[Line]) -> list[str]:
    found = " | ".join(f"found@{k}" for k in COVERAGE_FOUND_AT)
    out = [
        f"| source | asked by | figures | held | {found} | exact |",
        "|---" * (5 + len(COVERAGE_FOUND_AT)) + "|",
    ]
    for line in lines:
        cells = [line.source, line.phrasing, str(line.figures), _rate(line.held, line.figures)]
        cells += [_rate(line.found[k], line.held) for k in COVERAGE_FOUND_AT]
        cells.append("—" if line.exact is None else _rate(line.exact, line.held))
        out.append("| " + " | ".join(cells) + " |")
    return out


def to_markdown(scored: Sequence[Scored], left: Sequence[Left], header: Mapping[str, str]) -> str:
    """The report: the headline (the frozen fundamentals), every source, what was left out and
    why, and each fundamentals and tables miss."""
    lines = summarise(scored)
    out = ["# coverage", ""]
    out += [f"- {name}: {value}" for name, value in header.items()]
    out += [
        "",
        "Found and exact are out of the figures held. Exact is measured where a figure is "
        "matched by concept; matched by value, found is exact by construction (—).",
        "",
        "## Headline: the frozen fundamentals",
        "",
        *_table([line for line in lines if line.source == "fundamentals"]),
        "",
        "## Every source",
        "",
        *_table(lines),
    ]
    reasons: dict[tuple[str, str], int] = {}
    for item in left:
        reasons[(item.source, item.reason)] = reasons.get((item.source, item.reason), 0) + 1
    if reasons:
        out += ["", "## Left out", "", "| source | why | figures |", "|---|---|---|"]
        out += [f"| {s} | {r} | {n} |" for (s, r), n in sorted(reasons.items())]
    misses = [
        r
        for r in scored
        if r.source in ("fundamentals", "tables") and (r.rank is None or r.exact is False)
    ]
    if misses:
        out += ["", "## Misses: fundamentals and tables", ""]
        out += [f"- `{r.id}` [{r.phrasing}] — {r.detail}" for r in misses]
    return "\n".join(out) + "\n"


def write_report(
    scored: Sequence[Scored],
    left: Sequence[Left],
    header: Mapping[str, str],
    now: datetime,
    directory: Path = REPORTS,
) -> Path:
    """The report as `coverage-<time>.md`, every scored row beside it as `.rows.jsonl`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"coverage-{now:%Y%m%d-%H%M}.md"
    path.write_text(to_markdown(scored, left, header), encoding="utf-8")
    rows = "".join(json.dumps(asdict(row)) + "\n" for row in scored)
    path.with_suffix(".rows.jsonl").write_text(rows, encoding="utf-8")
    return path
