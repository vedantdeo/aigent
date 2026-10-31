"""The facts database in SQLite: its schema, and the build's writes. A rebuild keeps each XBRL
fact's row while its value holds, so what points at a fact survives it."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aigent.facts.taxonomy import Labels
from aigent.facts.xbrl import Fact, label_of

FACTS_DB = Path(__file__).resolve().parents[3] / "corpus" / ".cache" / "facts.sqlite"

XBRL_TIERS = ("xbrl-ai", "xbrl-pdf", "xbrl")  # as loaded, its document agreeing, a person's check
TIERS = (*XBRL_TIERS, "pdf-cell-ai", "pdf-cell", "pdf-read-ai", "pdf-read", "derived-ai", "derived")
CHECKED_TIERS = ("xbrl-pdf", "xbrl", "pdf-cell", "pdf-read", "derived")  # trusted; name a checker

_TIER_LIST = ", ".join(f"'{t}'" for t in TIERS)
_CHECKED_LIST = ", ".join(f"'{t}'" for t in CHECKED_TIERS)
_XBRL_LIST = ", ".join(f"'{t}'" for t in XBRL_TIERS)

SCHEMA_VERSION = 3  # 2: an xbrl row may carry verified_by; 3: xbrl-ai and xbrl-pdf
_FACTS = f"""
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    filing_id INTEGER REFERENCES filings(id),
    ticker TEXT NOT NULL,
    basis TEXT NOT NULL CHECK (basis IN ('consolidated', 'standalone')),
    concept TEXT NOT NULL,
    label TEXT NOT NULL,
    period_kind TEXT NOT NULL CHECK (period_kind IN ('duration', 'instant')),
    period_start TEXT,
    period_end TEXT NOT NULL,
    segment TEXT,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    decimals INTEGER,
    printed TEXT,
    tier TEXT NOT NULL CHECK (tier IN ({_TIER_LIST})),
    page INTEGER,
    chunk_id TEXT,
    row_index INTEGER,
    passage TEXT,
    formula TEXT,
    written_by TEXT NOT NULL,
    written_at TEXT NOT NULL,
    run_id TEXT NOT NULL,
    verified_by TEXT,
    verified_at TEXT,
    CHECK ((period_kind = 'instant') = (period_start IS NULL)),
    CHECK ((verified_by IS NULL) = (verified_at IS NULL)),
    CHECK ((tier IN ({_CHECKED_LIST})) = (verified_by IS NOT NULL))
);
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS filings (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    fy INTEGER NOT NULL,
    period_end TEXT NOT NULL,
    basis TEXT NOT NULL CHECK (basis IN ('consolidated', 'standalone')),
    source TEXT NOT NULL CHECK (source IN ('xbrl', 'pdf')),
    origin TEXT NOT NULL UNIQUE,
    sha256 TEXT,
    fetched TEXT
);
{_FACTS}CREATE UNIQUE INDEX IF NOT EXISTS facts_xbrl_once ON facts (
    filing_id, concept, period_kind, COALESCE(period_start, ''), period_end,
    COALESCE(segment, ''), unit
) WHERE tier IN ({_XBRL_LIST});
CREATE INDEX IF NOT EXISTS facts_lookup ON facts (ticker, basis, concept, period_end);
CREATE TABLE IF NOT EXISTS derived_from (
    fact_id INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    input_fact_id INTEGER NOT NULL REFERENCES facts(id),
    PRIMARY KEY (fact_id, input_fact_id)
);
CREATE TABLE IF NOT EXISTS concepts (
    concept TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    statement TEXT,
    kind TEXT NOT NULL CHECK (kind IN ('xbrl', 'row', 'read')),
    xbrl_concept TEXT,
    mapping_checked INTEGER NOT NULL DEFAULT 0,
    vector_key TEXT
);
CREATE TABLE IF NOT EXISTS concept_labels (
    concept TEXT NOT NULL REFERENCES concepts(concept),
    label TEXT NOT NULL,
    PRIMARY KEY (concept, label)
);
CREATE TABLE IF NOT EXISTS lookups (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    asked_by TEXT NOT NULL,
    asked_at TEXT NOT NULL,
    ticker TEXT,
    basis TEXT,
    period TEXT,
    what TEXT NOT NULL,
    concept TEXT,
    found INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class FilingRow:
    ticker: str
    fy: int
    period_end: date
    basis: str
    source: str
    origin: str
    sha256: str | None
    fetched: str | None


@dataclass(frozen=True)
class Writer:
    """Who writes a batch of rows, when, and in which run."""

    written_by: str
    written_at: str
    run_id: str


@dataclass
class Synced:
    """What syncing a filing's XBRL facts did."""

    added: int = 0
    kept: int = 0
    changed: int = 0
    removed: int = 0


def connect(path: Path = FACTS_DB) -> sqlite3.Connection:
    """The database, its schema created if missing and migrated if older; WAL, since runs write
    while others read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        _rebuild_facts(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# An `xbrl` row from before version 3, by its check: none, its document's alone, or a person's.
_XBRL_TIER_OF = """CASE WHEN tier != 'xbrl' THEN tier
    WHEN verified_by IS NULL THEN 'xbrl-ai'
    WHEN verified_by LIKE 'pdf:%' AND instr(verified_by, '+') = 0 THEN 'xbrl-pdf'
    ELSE 'xbrl' END"""


def _rebuild_facts(conn: sqlite3.Connection) -> None:
    """Re-creates `facts` under the current constraints, every row and id kept and an old `xbrl`
    row given its tier; SQLite cannot alter a CHECK in place. Run with foreign keys off, as
    SQLite's procedure requires."""
    columns = [row[1] for row in conn.execute("PRAGMA table_info(facts)")]
    selected = ", ".join(_XBRL_TIER_OF if c == "tier" else c for c in columns)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(
        "BEGIN;"
        + _FACTS.replace("IF NOT EXISTS facts", "facts_next")
        + f"INSERT INTO facts_next ({', '.join(columns)}) SELECT {selected} FROM facts;"
        "DROP TABLE facts;"
        "ALTER TABLE facts_next RENAME TO facts;"
        + SCHEMA
        + f"PRAGMA user_version = {SCHEMA_VERSION};"
        "COMMIT;"
    )


def upsert_filing(conn: sqlite3.Connection, row: FilingRow) -> int:
    """The filing's id, its row added or updated by origin."""
    conn.execute(
        """INSERT INTO filings (ticker, fy, period_end, basis, source, origin, sha256, fetched)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (origin) DO UPDATE
           SET sha256 = excluded.sha256, fetched = excluded.fetched""",
        (
            row.ticker,
            row.fy,
            row.period_end.isoformat(),
            row.basis,
            row.source,
            row.origin,
            row.sha256,
            row.fetched,
        ),
    )
    found = conn.execute("SELECT id FROM filings WHERE origin = ?", (row.origin,)).fetchone()
    return int(found[0])


def _fact_key(
    concept: str, kind: str, start: str | None, end: str, segment: str | None, unit: str
) -> tuple[str, str, str, str, str, str]:
    return (concept, kind, start or "", end, segment or "", unit)


def sync_xbrl_facts(
    conn: sqlite3.Connection,
    filing_id: int,
    ticker: str,
    basis: str,
    facts: Iterable[Fact],
    writer: Writer,
) -> Synced:
    """Makes the filing's XBRL rows equal `facts`: a new fact is `xbrl-ai`, an unchanged one keeps
    its row and tier, a changed value is updated, re-attributed and `xbrl-ai` again, unchecked, and
    a fact no longer loaded is deleted."""
    held: dict[tuple[str, str, str, str, str, str], tuple[int, float, int | None]] = {}
    for fid, concept, kind, start, end, segment, unit, value, decimals in conn.execute(
        f"""SELECT id, concept, period_kind, period_start, period_end, segment, unit, value,
                  decimals
           FROM facts WHERE filing_id = ? AND tier IN ({_XBRL_LIST})""",
        (filing_id,),
    ):
        held[_fact_key(concept, kind, start, end, segment, unit)] = (fid, value, decimals)
    out = Synced()
    for fact in facts:
        start = fact.period_start.isoformat() if fact.period_start else None
        end = fact.period_end.isoformat()
        key = _fact_key(fact.concept, fact.period_kind, start, end, fact.segment, fact.unit)
        prior = held.pop(key, None)
        if prior is not None and (prior[1], prior[2]) == (fact.value, fact.decimals):
            out.kept += 1
            continue
        if prior is not None:
            conn.execute(  # a revised value was never checked, so it is xbrl-ai again
                """UPDATE facts SET value = ?, decimals = ?, written_by = ?, written_at = ?,
                          run_id = ?, tier = 'xbrl-ai', verified_by = NULL, verified_at = NULL
                   WHERE id = ?""",
                (
                    fact.value,
                    fact.decimals,
                    writer.written_by,
                    writer.written_at,
                    writer.run_id,
                    prior[0],
                ),
            )
            out.changed += 1
            continue
        conn.execute(
            """INSERT INTO facts (filing_id, ticker, basis, concept, label, period_kind,
                      period_start, period_end, segment, value, unit, decimals, tier,
                      written_by, written_at, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'xbrl-ai', ?, ?, ?)""",
            (
                filing_id,
                ticker,
                basis,
                fact.concept,
                label_of(fact.concept),
                fact.period_kind,
                start,
                end,
                fact.segment,
                fact.value,
                fact.unit,
                fact.decimals,
                writer.written_by,
                writer.written_at,
                writer.run_id,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO concepts (concept, label, kind) VALUES (?, ?, 'xbrl')",
            (fact.concept, label_of(fact.concept)),
        )
        out.added += 1
    for fid, _, _ in held.values():
        conn.execute("DELETE FROM facts WHERE id = ?", (fid,))
        out.removed += 1
    return out


def apply_labels(conn: sqlite3.Connection, labels: Mapping[str, Labels]) -> list[str]:
    """Each XBRL concept's standard label onto it and its facts, its other labels into
    `concept_labels`; the concepts no package labels, which keep their split element name."""
    unlabelled: list[str] = []
    for (concept,) in conn.execute(
        "SELECT concept FROM concepts WHERE kind = 'xbrl' ORDER BY concept"
    ).fetchall():
        found = labels.get(concept)
        if found is None:
            unlabelled.append(concept)
            continue
        conn.execute("UPDATE concepts SET label = ? WHERE concept = ?", (found.standard, concept))
        conn.execute(
            f"UPDATE facts SET label = ? WHERE concept = ? AND tier IN ({_XBRL_LIST})",
            (found.standard, concept),
        )
        conn.execute("DELETE FROM concept_labels WHERE concept = ?", (concept,))
        conn.executemany(
            "INSERT INTO concept_labels (concept, label) VALUES (?, ?)",
            [(concept, alias) for alias in sorted(found.aliases)],
        )
    return unlabelled
