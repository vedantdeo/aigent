"""The facts database in SQLite: its schema, the build's writes, and an agent's derived facts,
derived concepts and formula proposals. A rebuild keeps each XBRL fact's row while its value
holds, so what points at a fact survives it."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aigent.facts.cfg_facts import COMPANY_SCOPES, SCOPES
from aigent.facts.taxonomy import Labels
from aigent.facts.xbrl import Fact, label_of

FACTS_DB = Path(__file__).resolve().parents[3] / "corpus" / ".cache" / "facts.sqlite"

XBRL_TIERS = ("xbrl-ai", "xbrl-filed", "xbrl")  # as loaded, its filed document agreeing, a person
TIERS = (
    *XBRL_TIERS,
    *("filed-cell-ai", "filed-cell", "filed-read-ai", "filed-read", "derived-ai", "derived"),
)
CHECKED_TIERS = ("xbrl-filed", "xbrl", "filed-cell", "filed-read", "derived")  # name a checker
FORMULA_PROPOSAL_TIERS = ("formula-ai", "formula")  # an agent's, accepted into the library
DERIVED_CONCEPT_TIERS = (
    "derived-concept-ai",
    "derived-concept",
)  # defined by an agent or seeded unchecked; a person's check

_TIER_LIST = ", ".join(f"'{t}'" for t in TIERS)
_CHECKED_LIST = ", ".join(f"'{t}'" for t in CHECKED_TIERS)
_XBRL_LIST = ", ".join(f"'{t}'" for t in XBRL_TIERS)

SCHEMA_VERSION = 4  # 2: xbrl may carry verified_by; 3: xbrl-ai, xbrl-pdf; 4: filed, not pdf
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

ANY_SCOPE = "*"


def scopes_of(spec: str) -> tuple[str, ...]:
    """The scopes a derived concept's scope names: `*` for all, else a comma-separated list."""
    if spec.strip() == ANY_SCOPE:
        return SCOPES
    return tuple(sorted({part.strip() for part in spec.split(",") if part.strip()}))


def scope_refusal(spec: str) -> str | None:
    """Why `spec` cannot be a scope, or None: `*`, or known scopes separated by commas."""
    named = scopes_of(spec)
    unknown = [x for x in named if x not in SCOPES]
    if not named or unknown or (ANY_SCOPE in spec and spec.strip() != ANY_SCOPE):
        return f"scope is {ANY_SCOPE} or a comma-separated list of {', '.join(SCOPES)}"
    return None


def normal_scope(spec: str) -> str:
    """`*` when a spec names every scope, else its scopes sorted and joined by commas."""
    named = scopes_of(spec)
    return ANY_SCOPE if set(named) == set(SCOPES) else ",".join(named)


def covers(spec: str, scope: str) -> bool:
    return scope in scopes_of(spec)


_DERIVED_CONCEPTS = """CREATE TABLE IF NOT EXISTS derived_concepts (
    name TEXT NOT NULL,
    scope TEXT NOT NULL,
    label TEXT NOT NULL,
    formula TEXT NOT NULL,
    bindings TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('derived-concept-ai', 'derived-concept')),
    question TEXT,
    written_by TEXT NOT NULL,
    written_at TEXT NOT NULL,
    run_id TEXT NOT NULL,
    verified_by TEXT,
    verified_at TEXT,
    PRIMARY KEY (name, scope),
    CHECK ((verified_by IS NULL) = (verified_at IS NULL)),
    CHECK ((tier = 'derived-concept') = (verified_by IS NOT NULL))
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
CREATE TABLE IF NOT EXISTS formula_proposals (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    expression TEXT NOT NULL,
    variables TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('formula-ai', 'formula')),
    question TEXT,
    written_by TEXT NOT NULL,
    written_at TEXT NOT NULL,
    run_id TEXT NOT NULL,
    verified_by TEXT,
    verified_at TEXT,
    UNIQUE (name, expression),
    CHECK ((verified_by IS NULL) = (verified_at IS NULL)),
    CHECK ((tier = 'formula') = (verified_by IS NOT NULL))
);
{_DERIVED_CONCEPTS}"""


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


@dataclass(frozen=True)
class DerivedRow:
    """A figure an agent computed from checked facts, as its `facts` row will hold it."""

    ticker: str
    basis: str
    concept: str
    label: str
    period_start: str | None
    period_end: str
    segment: str | None
    value: float
    unit: str
    formula: str


@dataclass
class Synced:
    """What syncing a filing's XBRL facts did."""

    added: int = 0
    kept: int = 0
    changed: int = 0
    removed: int = 0
    derived_dropped: int = 0  # derived facts computed from a changed or removed fact


# Tables renamed since: old name, new name, and each old tier's new name.
_BEFORE_SCOPES = "banking"  # every derived concept written before scopes came from a bank question
_RENAMED: tuple[tuple[str, str, dict[str, str], dict[str, str]], ...] = (
    (
        "metrics",
        "derived_concepts",
        {"metric": "derived-concept", "metric-ai": "derived-concept-ai"},
        {"scope": f"'{_BEFORE_SCOPES}'"},
    ),
    ("formulas", "formula_proposals", {}, {}),
)


def _move_renamed_tables(conn: sqlite3.Connection) -> None:
    """Moves each older table's rows into its new name, tiers renamed and missing columns added,
    and drops it; first gives a derived concepts table from before scopes its scope."""
    _add_scope(conn)
    for old, new, tiers, added in _RENAMED:
        held = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (old,))
        if held.fetchone() is None:
            continue
        columns = [str(r[1]) for r in conn.execute(f"PRAGMA table_info({old})")]
        tiered = " ".join(f"WHEN '{a}' THEN '{b}'" for a, b in tiers.items())
        selected = [
            f"CASE tier {tiered} ELSE tier END" if c == "tier" and tiers else c for c in columns
        ]
        names = ", ".join([*columns, *added])
        values = ", ".join([*selected, *added.values()])
        with conn:
            conn.execute(f"INSERT OR IGNORE INTO {new} ({names}) SELECT {values} FROM {old}")
            conn.execute(f"DROP TABLE {old}")


def _add_scope(conn: sqlite3.Connection) -> None:
    """Rebuilds a derived concepts table from before scopes (every row in `_BEFORE_SCOPES`) or
    from when the schema checked a scope against a fixed list."""
    columns = [str(r[1]) for r in conn.execute("PRAGMA table_info(derived_concepts)")]
    made = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'derived_concepts'"
    ).fetchone()
    if not columns or ("scope" in columns and "scope IN" not in str(made[0])):
        return
    listed = ", ".join(c for c in columns if c != "scope")
    scope = "scope" if "scope" in columns else f"'{_BEFORE_SCOPES}'"
    with conn:
        conn.execute("ALTER TABLE derived_concepts RENAME TO derived_concepts_before")
        conn.execute(_DERIVED_CONCEPTS)
        conn.execute(
            f"""INSERT INTO derived_concepts ({listed}, scope)
                SELECT {listed}, {scope} FROM derived_concepts_before"""
        )
        conn.execute("DROP TABLE derived_concepts_before")


def connect(path: Path = FACTS_DB) -> sqlite3.Connection:
    """The database, its schema created if missing and migrated if older; WAL, since runs write
    while others read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        _rebuild_facts(conn)
    _move_renamed_tables(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# A row's tier from an older version: an `xbrl` row from before 3 by its check (none, its
# document's alone, or a person's), and a `pdf` tier from before 4 renamed `filed`.
_TIER_OF = """CASE
    WHEN tier = 'xbrl' AND verified_by IS NULL THEN 'xbrl-ai'
    WHEN tier = 'xbrl' AND verified_by LIKE 'pdf:%' AND instr(verified_by, '+') = 0
        THEN 'xbrl-filed'
    WHEN tier = 'xbrl-pdf' THEN 'xbrl-filed'
    WHEN tier LIKE 'pdf-%' THEN 'filed-' || substr(tier, 5)
    ELSE tier END"""

# A document's stamp from before 4, `pdf:<path>`, with its origin: every PDF came from NSE and
# every HTML copy from an SEC 6-K.
_VERIFIED_BY_OF = """CASE
    WHEN verified_by NOT LIKE 'pdf:%' THEN verified_by
    WHEN instr(verified_by, '.pdf') > 0 THEN 'nse-pdf:' || substr(verified_by, 5)
    ELSE 'sec-6k-html:' || substr(verified_by, 5) END"""
_MIGRATED_AS = {"tier": _TIER_OF, "verified_by": _VERIFIED_BY_OF}


def _rebuild_facts(conn: sqlite3.Connection) -> None:
    """Re-creates `facts` under the current constraints, every row and id kept and an old row given
    its tier and stamp; SQLite cannot alter a CHECK in place. Run with foreign keys off, as
    SQLite's procedure requires."""
    columns: list[str] = [str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")]
    selected = ", ".join(_MIGRATED_AS.get(c, c) for c in columns)
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


def _drop_derived_from(conn: sqlite3.Connection, fact_id: int) -> int:
    """Deletes every derived fact computed from `fact_id`, its links going by the schema's cascade;
    how many."""
    derived = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT fact_id FROM derived_from WHERE input_fact_id = ?", (fact_id,)
        )
    ]
    for did in derived:
        conn.execute("DELETE FROM facts WHERE id = ?", (did,))
    return len(derived)


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
    a fact no longer loaded is deleted. A derived fact computed from a changed or deleted fact is
    deleted first; its formula stays."""
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
            out.derived_dropped += _drop_derived_from(conn, prior[0])
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
        out.derived_dropped += _drop_derived_from(conn, fid)
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


def write_formula_proposal(
    conn: sqlite3.Connection,
    name: str,
    expression: str,
    variables: Sequence[str],
    question: str | None,
    writer: Writer,
) -> tuple[int, bool]:
    """The formula's id, and whether this call wrote it: a new one goes in as `formula-ai`, and
    one already held under the same name and expression is reused as it stands."""
    held = conn.execute(
        "SELECT id FROM formula_proposals WHERE name = ? AND expression = ?", (name, expression)
    ).fetchone()
    if held is not None:
        return int(held[0]), False
    cursor = conn.execute(
        """INSERT INTO formula_proposals (name, expression, variables, tier, question, written_by,
                  written_at, run_id)
           VALUES (?, ?, ?, 'formula-ai', ?, ?, ?, ?)""",
        (
            name,
            expression,
            ", ".join(variables),
            question,
            writer.written_by,
            writer.written_at,
            writer.run_id,
        ),
    )
    return int(cursor.lastrowid or 0), True


def write_derived(
    conn: sqlite3.Connection,
    row: DerivedRow,
    inputs: Sequence[int],
    writer: Writer,
    checked_by: str | None = None,
) -> int:
    """Writes `row` as a fact linked to the facts it was computed from, and returns its id:
    `derived` naming `checked_by` (the concept check it rests on) when given, else `derived-ai`.
    ValueError, writing nothing, if an input is missing or in no checked tier."""
    marks = ", ".join("?" for _ in inputs)
    tiers = dict(conn.execute(f"SELECT id, tier FROM facts WHERE id IN ({marks})", tuple(inputs)))
    missing = [i for i in inputs if i not in tiers]
    unchecked = [i for i in inputs if i in tiers and tiers[i] not in CHECKED_TIERS]
    if not inputs or missing or unchecked:
        raise ValueError(f"derived from no input, missing {missing} or unchecked {unchecked}")
    cursor = conn.execute(
        """INSERT INTO facts (ticker, basis, concept, label, period_kind, period_start, period_end,
                  segment, value, unit, tier, formula, written_by, written_at, run_id,
                  verified_by, verified_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row.ticker,
            row.basis,
            row.concept,
            row.label,
            "instant" if row.period_start is None else "duration",
            row.period_start,
            row.period_end,
            row.segment,
            row.value,
            row.unit,
            "derived" if checked_by else "derived-ai",
            row.formula,
            writer.written_by,
            writer.written_at,
            writer.run_id,
            checked_by,
            writer.written_at if checked_by else None,
        ),
    )
    fact_id = int(cursor.lastrowid or 0)
    conn.executemany(
        "INSERT INTO derived_from (fact_id, input_fact_id) VALUES (?, ?)",
        [(fact_id, i) for i in dict.fromkeys(inputs)],
    )
    return fact_id


def write_derived_concept(
    conn: sqlite3.Connection,
    name: str,
    scope: str,
    label: str,
    formula: str,
    bindings: str,
    question: str | None,
    writer: Writer,
) -> bool:
    """Writes a derived concept as `derived-concept-ai` in `scope` (normalised); False, writing
    nothing, when one is already held by that name in any scope it names, whatever it says."""
    scope = normal_scope(scope)
    held = [
        r[0] for r in conn.execute("SELECT scope FROM derived_concepts WHERE name = ?", (name,))
    ]
    if any(set(scopes_of(h)) & set(scopes_of(scope)) for h in held):
        return False
    cursor = conn.execute(
        """INSERT OR IGNORE INTO derived_concepts (name, scope, label, formula, bindings, tier,
                  question, written_by, written_at, run_id)
           VALUES (?, ?, ?, ?, ?, 'derived-concept-ai', ?, ?, ?, ?)""",
        (
            name,
            scope,
            label,
            formula,
            bindings,
            question,
            writer.written_by,
            writer.written_at,
            writer.run_id,
        ),
    )
    return cursor.rowcount == 1


def concept_check(name: str, scope: str, person: str) -> str:
    """What a trusted derived fact names as its check: its concept and scope, and who checked."""
    return f"{name}[{scope}]@{person}"


def verify_derived_concept(
    conn: sqlite3.Connection, name: str, scope: str, person: str, at: str
) -> tuple[bool, int]:
    """Records `person`'s check of a derived concept's bindings, making it `derived-concept`, and
    makes `derived` every `derived-ai` fact computed with it whose inputs are all in checked
    tiers, for the companies of `scope`. Returns whether the check was new and how many facts it
    promoted. ValueError if no derived concept has that name in that scope or no person is named."""
    scope = normal_scope(scope)
    held = conn.execute(
        "SELECT tier FROM derived_concepts WHERE name = ? AND scope = ?", (name, scope)
    ).fetchone()
    if held is None:
        raise ValueError(f"no derived concept {name!r} in scope {scope!r}")
    if not person.strip():
        raise ValueError("say who checked it")
    with conn:
        cursor = conn.execute(
            """UPDATE derived_concepts SET tier = 'derived-concept', verified_by = ?,
                      verified_at = ?
               WHERE name = ? AND scope = ? AND tier = 'derived-concept-ai'""",
            (person.strip(), at, name, scope),
        )
        checked = conn.execute(
            "SELECT verified_by FROM derived_concepts WHERE name = ? AND scope = ?", (name, scope)
        ).fetchone()[0]
        marks = ", ".join("?" for _ in CHECKED_TIERS)
        tickers = [t for t, s in COMPANY_SCOPES.items() if covers(scope, s)]
        companies = ", ".join("?" for _ in tickers)
        promoted = conn.execute(
            f"""UPDATE facts SET tier = 'derived', verified_by = ?, verified_at = ?
                WHERE tier = 'derived-ai' AND concept = ? AND ticker IN ({companies})
                  AND NOT EXISTS (
                    SELECT 1 FROM derived_from d JOIN facts i ON i.id = d.input_fact_id
                    WHERE d.fact_id = facts.id AND i.tier NOT IN ({marks}))""",
            (concept_check(name, scope, checked), at, name, *tickers, *CHECKED_TIERS),
        )
    return cursor.rowcount == 1, promoted.rowcount
