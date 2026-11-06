"""Looking facts up: `what` matched to concepts by their labels, or a concept named exactly, fetched
by company, financial year, period, basis and segment; every lookup logged in `lookups`. Also the
time an agent names: a quarter's end and the length of the span ending there."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast, get_args

from aigent.facts.cfg_facts import FACTS_LOOKUP_K, FACTS_MATCH_DEPTH, FACTS_MATCH_METHOD
from aigent.facts.store import CHECKED_TIERS, connect
from aigent.retrieval.chunk import Chunk, Inventory
from aigent.retrieval.embed import Embedder
from aigent.retrieval.rank import Indexes, Ranker, build_ranker

Period = Literal["year", "Q1", "Q2", "Q3", "Q4", "H1", "9M"]
Basis = Literal["consolidated", "standalone"]
PERIODS: tuple[Period, ...] = get_args(Period)
BASES: tuple[Basis, ...] = get_args(Basis)
Quarter = Literal["Q1", "Q2", "Q3", "Q4"]
QUARTERS: tuple[Quarter, ...] = get_args(Quarter)
Length = Literal["3M", "6M", "9M", "12M"]
LENGTHS: tuple[Length, ...] = get_args(Length)
# Each period as the quarter it ends at and its length; a filing holds no other span.
_TIME_OF: dict[Period, tuple[Quarter, Length]] = {
    "Q1": ("Q1", "3M"),
    "Q2": ("Q2", "3M"),
    "Q3": ("Q3", "3M"),
    "Q4": ("Q4", "3M"),
    "H1": ("Q2", "6M"),
    "9M": ("Q3", "9M"),
    "year": ("Q4", "12M"),
}

# Each period of financial year Y as (first month, last month), months counted from April of Y-1.
_SPANS: dict[Period, tuple[int, int]] = {
    "year": (1, 12),
    "Q1": (1, 3),
    "Q2": (4, 6),
    "Q3": (7, 9),
    "Q4": (10, 12),
    "H1": (1, 6),
    "9M": (1, 9),
}
_LAST_DAY = {6: 30, 9: 30, 12: 31, 3: 31}


def as_period(
    year: int, quarter: str, length: str | None, snapshot: bool, name: str
) -> tuple[int, Period]:
    """A named time as a year and period: a snapshot as at `quarter`'s end, a flow the span of
    `length` ending then. ValueError for a span no filing holds, a snapshot given a length or a
    flow given none."""
    if snapshot and length is not None:
        raise ValueError(f"{name} is a snapshot, as at a quarter's end; give no period")
    if not snapshot and length is None:
        raise ValueError(f"{name} reads a span; give its period, 3M, 6M, 9M or 12M")
    wanted = (quarter, "3M" if length is None else length)
    for period, time in _TIME_OF.items():
        if time == wanted:
            return year, period
    ends = {t[1]: t[0] for t in _TIME_OF.values() if t[1] != "3M"}.get(str(length), "any quarter")
    raise ValueError(f"a {length} period ends at {ends} only; 3M ends at any quarter")


def as_time(period: Period) -> tuple[Quarter, Length]:
    """The quarter `period` ends at and its length."""
    return _TIME_OF[period]


def opening_of(year: int, period: Period) -> tuple[int, Period]:
    """The quarter whose end is the day before `period` of `year` begins: its opening balances."""
    first = _SPANS[period][0]
    if first == 1:
        return year - 1, "Q4"
    return year, cast(Period, f"Q{(first - 1) // 3}")


def instants(conn: sqlite3.Connection, concepts: Sequence[str]) -> set[str]:
    """Those of `concepts` filed mostly as balances as at a date."""
    rows = conn.execute(
        f"""SELECT concept FROM facts WHERE concept IN ({", ".join("?" for _ in concepts)})
            GROUP BY concept HAVING SUM(period_kind = 'instant') > SUM(period_kind = 'duration')""",
        tuple(concepts),
    )
    return {str(r[0]) for r in rows}


class NotFound(ValueError):
    """Nothing to answer with, and why, worded for the model that asked."""


def span(year: int, period: Period) -> tuple[date, date]:
    """The first and last day of `period` in the financial year ending in March of `year`."""
    first, last = _SPANS[period]
    start_month = (first + 2) % 12 + 1
    end_month = (last + 2) % 12 + 1
    start = date(year - 1 if start_month >= 4 else year, start_month, 1)
    end = date(year - 1 if end_month >= 4 else year, end_month, _LAST_DAY[end_month])
    return start, end


@dataclass(frozen=True)
class Found:
    """One fact as the tool returns it."""

    cite_as: str
    concept: str
    label: str
    value: float
    unit: str
    period_start: str | None
    period_end: str
    basis: str
    segment: str | None
    tier: str
    source: str

    def as_json(self) -> dict[str, object]:
        period: dict[str, str] = (
            {"as_at": self.period_end}
            if self.period_start is None
            else {"from": self.period_start, "to": self.period_end}
        )
        return {
            "cite_as": self.cite_as,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "period": period,
            "basis": self.basis,
            "segment": self.segment,
            "tier": self.tier,
            "source": self.source,
        }


class ConceptIndex:
    """The concepts' labels, indexed to rank against the words of a question."""

    def __init__(
        self,
        texts: Mapping[str, str],
        embedder: Embedder | None = None,
        method: str = FACTS_MATCH_METHOD,
        *,
        cache: bool = False,
    ) -> None:
        chunks = tuple(
            Chunk(id=concept, text=text, doc_id="concepts", ordinal=i, start=0)
            for i, (concept, text) in enumerate(sorted(texts.items()))
        )
        inventory = Inventory("facts-concepts", chunks, {c.id: c for c in chunks})
        self.concepts = frozenset(texts)
        self._ranker: Ranker = build_ranker(method, Indexes.build(inventory, embedder, cache=cache))

    def match(self, what: str, k: int = FACTS_MATCH_DEPTH) -> list[str]:
        """The concepts closest to `what`, an exact concept name first."""
        ranked = [hit.chunk_id for hit in self._ranker.rank(what, k)]
        if what in self.concepts:
            return [what, *(c for c in ranked if c != what)][:k]
        return ranked


def concept_texts(conn: sqlite3.Connection) -> dict[str, str]:
    """Each concept's standard label, then its other labels, as the text it is matched by."""
    aliases: dict[str, list[str]] = {}
    for concept, label in conn.execute(
        "SELECT concept, label FROM concept_labels ORDER BY concept, label"
    ):
        aliases.setdefault(concept, []).append(label)
    return {
        concept: "; ".join([label, *aliases.get(concept, [])])
        for concept, label in conn.execute("SELECT concept, label FROM concepts ORDER BY concept")
    }


def segment_of(
    conn: sqlite3.Connection, ticker: str, basis: str, end: date, asked: str | None
) -> str | None:
    """The segment a company files that `asked` names: the same words, or the one that holds them;
    NotFound naming what it does file."""
    if asked is None or not asked.strip():
        return None
    held = [
        row[0]
        for row in conn.execute(
            """SELECT DISTINCT segment FROM facts WHERE ticker = ? AND basis = ? AND period_end = ?
               AND segment IS NOT NULL ORDER BY segment""",
            (ticker, basis, end.isoformat()),
        )
    ]
    wanted = asked.strip().casefold()
    exact = [s for s in held if s.casefold() == wanted]
    if exact:
        return exact[0]
    holding = [s for s in held if wanted in s.casefold()]
    if len(holding) == 1:
        return holding[0]
    named = "; ".join(held) if held else "none"
    raise NotFound(f"no single segment {asked!r} for {ticker} ({basis}); it files: {named}")


def find(
    conn: sqlite3.Connection,
    ticker: str,
    year: int,
    period: Period,
    basis: str,
    segment: str | None,
    concepts: Sequence[str],
    run_id: str,
    k: int = FACTS_LOOKUP_K,
) -> list[Found]:
    """Up to `k` facts, one per concept in the order given, for exactly this company, period, basis
    and segment, of any tier: a checked fact first, then this run's own; a period filed twice is
    answered from its own filing."""
    start, end = span(year, period)
    checked = ", ".join("?" for _ in CHECKED_TIERS)
    found: list[Found] = []
    for concept in concepts:
        row = conn.execute(
            f"""SELECT f.label, f.value, f.unit, f.period_start, f.period_end, f.tier, fl.origin,
                       fl.period_end
                FROM facts f LEFT JOIN filings fl ON fl.id = f.filing_id
                WHERE f.ticker = ? AND f.basis = ? AND f.concept = ?
                  AND COALESCE(f.segment, '') = ?
                  AND ((f.period_kind = 'duration' AND f.period_start = ? AND f.period_end = ?)
                       OR (f.period_kind = 'instant' AND f.period_end = ?))
                ORDER BY CASE WHEN f.tier IN ({checked}) THEN 0 WHEN f.run_id = ? THEN 1
                         ELSE 2 END,
                         fl.period_end
                LIMIT 1""",
            (
                ticker,
                basis,
                concept,
                segment or "",
                start.isoformat(),
                end.isoformat(),
                end.isoformat(),
                *CHECKED_TIERS,
                run_id,
            ),
        ).fetchone()
        if row is None:
            continue
        label, value, unit, period_start, period_end, tier, origin, filed = row
        cite = f"FACT:{ticker}:{year}:{basis}:{period}:{concept}"
        source = (
            f"NSE XBRL results filing, period ended {filed}: {origin}" if origin else "written back"
        )
        found.append(
            Found(
                cite_as=cite if segment is None else f"{cite}:{segment}",
                concept=concept,
                label=label,
                value=value,
                unit=unit,
                period_start=period_start,
                period_end=period_end,
                basis=basis,
                segment=segment,
                tier=tier,
                source=source,
            )
        )
        if len(found) == k:
            break
    return found


@dataclass
class Lookup:
    """Facts for a question's words, from the database at `db`, with the concept index built on
    first use; every call is logged as `asked_by` in run `run_id`."""

    db: Path
    index: Callable[[sqlite3.Connection], ConceptIndex]
    run_id: str
    asked_by: str
    clock: Callable[[], datetime] = lambda: datetime.now().astimezone()
    _index: ConceptIndex | None = field(default=None, init=False)
    _building: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __call__(
        self,
        ticker: str,
        year: int,
        what: str,
        basis: Basis = "consolidated",
        period: Period = "year",
        segment: str | None = None,
    ) -> list[Found]:
        if not self.db.exists():
            raise NotFound("the facts database is not built: python -m aigent.facts build")
        conn = connect(self.db)
        try:
            with self._building:
                if self._index is None:
                    self._index = self.index(conn)
            matched = self._index.match(what)
            found: list[Found] = []
            try:
                end = span(year, period)[1]
                named = segment_of(conn, ticker, basis, end, segment)
                found = find(conn, ticker, year, period, basis, named, matched, self.run_id)
            finally:
                self._log(conn, ticker, year, what, basis, period, found, matched)
        finally:
            conn.close()
        if not found:
            closest = "; ".join(_labels(self.db, matched[:3])) or "none"
            raise NotFound(
                f"{ticker} files nothing matching {what!r} for FY{year} {period} ({basis}"
                f"{'' if segment is None else ', segment ' + segment}). Closest labels filed: "
                f"{closest}. Ask in those words, or search the filings."
            )
        return found

    def candidates(self, ticker: str, what: str, k: int = FACTS_LOOKUP_K) -> list[Candidate]:
        """The concepts `ticker` files closest to `what`, an exact name first, at most `k`."""
        conn = connect(self.db)
        try:
            with self._building:
                if self._index is None:
                    self._index = self.index(conn)
            matched = self._index.match(what)
            filed = _filed(conn, ticker, matched)
            kept = [c for c in matched if c in filed][:k]
            balances = instants(conn, kept) if kept else set()
            labels = dict(
                conn.execute(
                    f"SELECT concept, label FROM concepts WHERE concept IN "
                    f"({', '.join('?' for _ in kept)})",
                    tuple(kept),
                ).fetchall()
            )
        finally:
            conn.close()
        return [Candidate(c, str(labels.get(c, c)), c in balances) for c in kept]

    def fact(
        self,
        ticker: str,
        concept: str,
        year: int,
        period: Period,
        basis: Basis = "consolidated",
        segment: str | None = None,
    ) -> list[Found]:
        """The fact of exactly `concept`, logged; NotFound when there is none for that period."""
        conn = connect(self.db)
        try:
            found: list[Found] = []
            try:
                named = segment_of(conn, ticker, basis, span(year, period)[1], segment)
                found = find(conn, ticker, year, period, basis, named, [concept], self.run_id)
            finally:
                self._log(conn, ticker, year, concept, basis, period, found, [concept])
        finally:
            conn.close()
        if not found:
            raise NotFound(
                f"{ticker} files no {concept} for FY{year} {period} ({basis}"
                f"{'' if segment is None else ', segment ' + segment}); try another time"
            )
        return found

    def snapshot(self, ticker: str, concept: str) -> bool | None:
        """Whether `concept` is a balance as at a date; None when `ticker` files no such concept."""
        conn = connect(self.db)
        try:
            if not _filed(conn, ticker, [concept]):
                return None
            return concept in instants(conn, [concept])
        finally:
            conn.close()

    def _log(
        self,
        conn: sqlite3.Connection,
        ticker: str,
        year: int,
        what: str,
        basis: str,
        period: str,
        found: Sequence[Found],
        matched: Sequence[str],
    ) -> None:
        concept = found[0].concept if found else (matched[0] if matched else None)
        with conn:
            conn.execute(
                """INSERT INTO lookups (run_id, asked_by, asked_at, ticker, basis, period, what,
                          concept, found)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.run_id,
                    self.asked_by,
                    self.clock().isoformat(timespec="seconds"),
                    ticker,
                    basis,
                    f"{year} {period}",
                    what,
                    concept,
                    int(bool(found)),
                ),
            )


@dataclass(frozen=True)
class Candidate:
    """A concept a company files, its label, and whether it is a balance as at a date."""

    concept: str
    label: str
    snapshot: bool


def _filed(conn: sqlite3.Connection, ticker: str, concepts: Sequence[str]) -> set[str]:
    rows = conn.execute(
        f"""SELECT DISTINCT concept FROM facts WHERE ticker = ?
            AND concept IN ({", ".join("?" for _ in concepts)})""",
        (ticker, *concepts),
    )
    return {str(r[0]) for r in rows}


def _labels(db: Path, concepts: Sequence[str]) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        by_concept = dict(
            conn.execute(
                f"SELECT concept, label FROM concepts WHERE concept IN "
                f"({', '.join('?' for _ in concepts)})",
                tuple(concepts),
            ).fetchall()
        )
    finally:
        conn.close()
    return [by_concept[c] for c in concepts if c in by_concept]


def concept_index(embedder: Embedder | None = None) -> Callable[[sqlite3.Connection], ConceptIndex]:
    """How `Lookup` builds its index: every concept's text from the database, ranked by
    `FACTS_MATCH_METHOD`, its vectors cached."""

    def build(conn: sqlite3.Connection) -> ConceptIndex:
        return ConceptIndex(concept_texts(conn), embedder, cache=embedder is None)

    return build
