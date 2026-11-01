"""The facts database's schema guards and the build's sync of a filing's XBRL rows."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from aigent.facts.store import (
    SCHEMA_VERSION,
    FilingRow,
    Writer,
    apply_labels,
    connect,
    sync_xbrl_facts,
    upsert_filing,
)
from aigent.facts.taxonomy import Labels
from aigent.facts.xbrl import Fact

WRITER = Writer("xbrl-loader@abc1234", "2026-10-22T12:00:00+05:30", "build-1")
REVENUE = Fact(
    "RevenueFromOperations", "duration", date(2024, 4, 1), date(2025, 3, 31), None, 100.0, "INR", -5
)
ASSETS = Fact("Assets", "instant", None, date(2025, 3, 31), None, 50.0, "INR", -5)


def _filing(conn: sqlite3.Connection, origin: str = "https://x/a.xml") -> int:
    return upsert_filing(
        conn,
        FilingRow("ITC", 2025, date(2025, 3, 31), "consolidated", "xbrl", origin, "s", "t"),
    )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return connect(tmp_path / "facts.sqlite")


def test_connect_creates_the_schema_in_wal_mode(conn: sqlite3.Connection) -> None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"concepts", "derived_from", "facts", "filings", "lookups"} <= tables
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_upsert_filing_returns_one_id_per_origin(conn: sqlite3.Connection) -> None:
    first, again, other = _filing(conn), _filing(conn), _filing(conn, "https://x/b.xml")
    assert first == again != other


def test_sync_adds_keeps_changes_and_removes(conn: sqlite3.Connection) -> None:
    fid = _filing(conn)
    added = sync_xbrl_facts(conn, fid, "ITC", "consolidated", [REVENUE, ASSETS], WRITER)
    ids = dict(conn.execute("SELECT concept, id FROM facts"))
    later = Writer("xbrl-loader@def5678", "2026-10-23T12:00:00+05:30", "build-2")

    kept = sync_xbrl_facts(conn, fid, "ITC", "consolidated", [REVENUE, ASSETS], later)
    assert dict(conn.execute("SELECT concept, id FROM facts")) == ids
    assert conn.execute("SELECT DISTINCT run_id FROM facts").fetchall() == [("build-1",)]

    changed = sync_xbrl_facts(
        conn, fid, "ITC", "consolidated", [replace(REVENUE, value=101.0)], later
    )
    row = conn.execute(
        "SELECT id, value, written_by, run_id FROM facts WHERE concept = 'RevenueFromOperations'"
    ).fetchone()
    assert row == (ids["RevenueFromOperations"], 101.0, "xbrl-loader@def5678", "build-2")
    assert conn.execute("SELECT count(*) FROM facts WHERE concept = 'Assets'").fetchone()[0] == 0

    assert (added.added, kept.kept, changed.changed, changed.removed) == (2, 2, 1, 1)


def test_sync_records_ticker_basis_label_tier_and_writer(conn: sqlite3.Connection) -> None:
    sync_xbrl_facts(conn, _filing(conn), "ITC", "consolidated", [REVENUE], WRITER)
    row = conn.execute(
        """SELECT ticker, basis, label, tier, period_start, period_end, written_by, written_at,
                  run_id, verified_by FROM facts"""
    ).fetchone()
    assert row == (
        "ITC",
        "consolidated",
        "Revenue from operations",
        "xbrl-ai",
        "2024-04-01",
        "2025-03-31",
        *(WRITER.written_by, WRITER.written_at, WRITER.run_id),
        None,
    )
    assert conn.execute("SELECT concept, label, kind FROM concepts").fetchall() == [
        ("RevenueFromOperations", "Revenue from operations", "xbrl")
    ]


def test_sync_touches_only_its_own_filing(conn: sqlite3.Connection) -> None:
    a, b = _filing(conn), _filing(conn, "https://x/b.xml")
    sync_xbrl_facts(conn, a, "ITC", "consolidated", [REVENUE], WRITER)
    sync_xbrl_facts(conn, b, "ITC", "consolidated", [], WRITER)
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1


_ROW = {
    "ticker": "ITC",
    "basis": "consolidated",
    "concept": "C",
    "label": "C",
    "period_kind": "duration",
    "period_start": "2024-04-01",
    "period_end": "2025-03-31",
    "value": 1.0,
    "unit": "INR",
    "tier": "pdf-cell-ai",
    "written_by": "w",
    "written_at": "t",
    "run_id": "r",
    "verified_by": None,
    "verified_at": None,
}


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"period_kind": "instant"}, id="a balance with a start date"),
        pytest.param({"period_start": None}, id="a duration with no start date"),
        pytest.param({"tier": "pdf-cell"}, id="a checked tier nobody checked"),
        pytest.param(
            {"verified_by": "vedant", "verified_at": "t"}, id="an -ai tier marked checked"
        ),
        pytest.param({"tier": "pdf-cell", "verified_by": "vedant"}, id="checked with no time"),
        pytest.param({"tier": "xbrl"}, id="a person's xbrl tier with nobody named"),
        pytest.param({"tier": "xbrl-pdf"}, id="a document's agreement with no document named"),
        pytest.param(
            {"tier": "xbrl-ai", "verified_by": "pdf:x.htm", "verified_at": "t"},
            id="an unseen xbrl fact marked checked",
        ),
        pytest.param({"tier": "xbrl-pdf", "verified_by": "pdf:x.htm"}, id="xbrl checked, no time"),
        pytest.param({"tier": "xbrl-pdf", "verified_at": "t"}, id="a time and no checker"),
        pytest.param({"tier": "guess"}, id="a tier that does not exist"),
        pytest.param({"basis": "group"}, id="a basis that does not exist"),
    ],
)
def test_the_schema_refuses_a_row_that_breaks_a_rule(
    conn: sqlite3.Connection, change: dict[str, object]
) -> None:
    row = _ROW | change
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"INSERT INTO facts ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
            tuple(row.values()),
        )


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({}, id="an unchecked cell"),
        pytest.param(
            {"tier": "pdf-read", "verified_by": "vedant", "verified_at": "t"}, id="a checked read"
        ),
        pytest.param({"period_kind": "instant", "period_start": None}, id="a balance"),
        pytest.param({"tier": "xbrl-ai"}, id="an xbrl fact nobody has seen"),
        pytest.param(
            {"tier": "xbrl-pdf", "verified_by": "pdf:results/X-2025-03-31.pdf", "verified_at": "t"},
            id="an xbrl fact its document agrees with",
        ),
        pytest.param(
            {"tier": "xbrl", "verified_by": "vedant", "verified_at": "t"},
            id="an xbrl fact a person signed off",
        ),
    ],
)
def test_the_schema_takes_a_row_that_keeps_the_rules(
    conn: sqlite3.Connection, change: dict[str, object]
) -> None:
    row = _ROW | change
    conn.execute(
        f"INSERT INTO facts ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
        tuple(row.values()),
    )


def test_one_filing_holds_an_xbrl_fact_once(conn: sqlite3.Connection) -> None:
    fid = _filing(conn)
    sync_xbrl_facts(conn, fid, "ITC", "consolidated", [ASSETS], WRITER)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO facts (filing_id, ticker, basis, concept, label, period_kind,
                      period_end, value, unit, tier, written_by, written_at, run_id)
               SELECT filing_id, ticker, basis, concept, label, period_kind, period_end, 7.0,
                      unit, tier, written_by, written_at, 'build-2' FROM facts"""
        )


def test_labels_go_onto_concepts_and_facts_and_aliases_are_replaced(
    conn: sqlite3.Connection,
) -> None:
    sync_xbrl_facts(conn, _filing(conn), "ITC", "consolidated", [REVENUE, ASSETS], WRITER)
    apply_labels(conn, {"RevenueFromOperations": Labels("Revenue (official)", {"Old alias"})})
    unlabelled = apply_labels(
        conn, {"RevenueFromOperations": Labels("Revenue (official)", {"Sales", "Turnover"})}
    )
    assert unlabelled == ["Assets"]
    assert conn.execute("SELECT concept, label FROM concepts ORDER BY concept").fetchall() == [
        ("Assets", "Assets"),
        ("RevenueFromOperations", "Revenue (official)"),
    ]
    assert conn.execute(
        "SELECT label FROM facts WHERE concept = 'RevenueFromOperations'"
    ).fetchall() == [("Revenue (official)",)]
    assert conn.execute("SELECT concept, label FROM concept_labels ORDER BY label").fetchall() == [
        ("RevenueFromOperations", "Sales"),
        ("RevenueFromOperations", "Turnover"),
    ]


_OLD_FACTS = """CREATE TABLE facts (
    id INTEGER PRIMARY KEY, filing_id INTEGER REFERENCES filings(id), ticker TEXT NOT NULL,
    basis TEXT NOT NULL, concept TEXT NOT NULL, label TEXT NOT NULL, period_kind TEXT NOT NULL,
    period_start TEXT, period_end TEXT NOT NULL, segment TEXT, value REAL NOT NULL,
    unit TEXT NOT NULL, decimals INTEGER, printed TEXT, tier TEXT NOT NULL, page INTEGER,
    chunk_id TEXT, row_index INTEGER, passage TEXT, formula TEXT, written_by TEXT NOT NULL,
    written_at TEXT NOT NULL, run_id TEXT NOT NULL, verified_by TEXT, verified_at TEXT);
CREATE TABLE filings (id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, fy INTEGER NOT NULL,
    period_end TEXT NOT NULL, basis TEXT NOT NULL, source TEXT NOT NULL,
    origin TEXT NOT NULL UNIQUE, sha256 TEXT, fetched TEXT);
CREATE TABLE derived_from (fact_id INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    input_fact_id INTEGER NOT NULL REFERENCES facts(id), PRIMARY KEY (fact_id, input_fact_id));
"""

# An older database's rows, by id: (tier, verified_by) before, the tier after.
_MIGRATED = [
    (7, "xbrl", None, "xbrl-ai"),
    (8, "xbrl", "pdf:results/ITC-2025-03-31.pdf", "xbrl-pdf"),
    (9, "xbrl", "vedant", "xbrl"),
    (10, "xbrl", "pdf:results/ITC-2025-03-31.pdf+vedant", "xbrl"),
    (11, "pdf-cell", "vedant", "pdf-cell"),
    (12, "pdf-read-ai", None, "pdf-read-ai"),
]


@pytest.fixture
def old_db(tmp_path: Path) -> Path:
    """A database from before schema version 3, one row per way an old row can be checked."""
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript(_OLD_FACTS)
    old.execute(
        "INSERT INTO filings VALUES (1, 'ITC', 2025, '2025-03-31', 'c', 'xbrl', 'u', 's', 't')"
    )
    for fid, tier, by, _ in _MIGRATED:
        old.execute(
            """INSERT INTO facts (id, filing_id, ticker, basis, concept, label, period_kind,
                      period_end, value, unit, tier, written_by, written_at, run_id, verified_by,
                      verified_at)
               VALUES (?, 1, 'ITC', 'consolidated', ?, ?, 'instant', '2025-03-31', 1.0, 'INR',
                       ?, 'w', 't', 'r', ?, ?)""",
            (fid, f"C{fid}", f"C{fid}", tier, by, None if by is None else "t"),
        )
    old.execute("INSERT INTO derived_from VALUES (8, 7)")
    old.commit()
    old.close()
    return path


@pytest.mark.parametrize(
    ("fid", "before", "after"),
    [
        pytest.param(fid, (tier, by), after, id=f"{tier} checked by {by}")
        for fid, tier, by, after in _MIGRATED
    ],
)
def test_an_older_database_gives_each_row_its_tier(
    old_db: Path, fid: int, before: tuple[str, str | None], after: str
) -> None:
    conn = connect(old_db)
    tier, by = conn.execute("SELECT tier, verified_by FROM facts WHERE id = ?", (fid,)).fetchone()
    assert (tier, by) == (after, before[1]), (tier, by)


def test_an_older_database_keeps_its_rows_links_and_indexes(old_db: Path) -> None:
    conn = connect(old_db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert [r[0] for r in conn.execute("SELECT id FROM facts ORDER BY id")] == [
        fid for fid, *_ in _MIGRATED
    ]
    assert conn.execute("SELECT * FROM derived_from").fetchall() == [(8, 7)]
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"facts_xbrl_once", "facts_lookup"} <= names
    with pytest.raises(sqlite3.IntegrityError):  # the new constraints hold
        conn.execute("UPDATE facts SET verified_by = NULL, verified_at = NULL WHERE id = 8")
    conn.close()
    again = connect(old_db)  # a current database is not rebuilt
    assert again.execute("SELECT count(*) FROM facts").fetchone()[0] == len(_MIGRATED)


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        pytest.param(
            100.0, ("xbrl-pdf", "pdf:results/ITC-2025-03-31.pdf", "t"), id="an unchanged value"
        ),
        pytest.param(101.0, ("xbrl-ai", None, None), id="a revised value"),
    ],
)
def test_a_rebuild_keeps_a_check_only_while_the_value_holds(
    conn: sqlite3.Connection, value: float, kept: tuple[str, str | None, str | None]
) -> None:
    fid = _filing(conn)
    sync_xbrl_facts(conn, fid, "ITC", "consolidated", [REVENUE], WRITER)
    conn.execute(
        """UPDATE facts SET tier = 'xbrl-pdf', verified_by = 'pdf:results/ITC-2025-03-31.pdf',
                  verified_at = 't'"""
    )
    sync_xbrl_facts(conn, fid, "ITC", "consolidated", [replace(REVENUE, value=value)], WRITER)
    row = conn.execute("SELECT tier, verified_by, verified_at FROM facts").fetchone()
    assert row == kept, row


@pytest.mark.parametrize(
    ("tier", "checker"),
    [
        pytest.param("xbrl-pdf", "pdf:results/ITC-2025-03-31.pdf", id="its document agrees"),
        pytest.param("xbrl", "vedant", id="a person checked it"),
    ],
)
def test_a_checked_xbrl_fact_is_labelled_and_held_once(
    conn: sqlite3.Connection, tier: str, checker: str
) -> None:
    sync_xbrl_facts(conn, _filing(conn), "ITC", "consolidated", [REVENUE], WRITER)
    conn.execute("UPDATE facts SET tier = ?, verified_by = ?, verified_at = 't'", (tier, checker))
    apply_labels(conn, {"RevenueFromOperations": Labels("Revenue (official)", set())})
    assert conn.execute("SELECT label FROM facts").fetchall() == [("Revenue (official)",)]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO facts (filing_id, ticker, basis, concept, label, period_kind,
                      period_start, period_end, value, unit, tier, written_by, written_at,
                      run_id, verified_by, verified_at)
               SELECT filing_id, ticker, basis, concept, label, period_kind, period_start,
                      period_end, 7.0, unit, tier, written_by, written_at, 'build-2',
                      verified_by, verified_at FROM facts"""
        )
