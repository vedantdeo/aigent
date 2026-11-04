"""The facts database's schema guards, the build's sync of a filing's XBRL rows, and an agent's
derived facts and formulas."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from aigent.facts.store import (
    SCHEMA_VERSION,
    DerivedRow,
    FilingRow,
    Writer,
    apply_labels,
    connect,
    covers,
    normal_scope,
    scope_refusal,
    scopes_of,
    sync_xbrl_facts,
    upsert_filing,
    verify_derived_concept,
    write_derived,
    write_derived_concept,
    write_formula_proposal,
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
    "tier": "filed-cell-ai",
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
        pytest.param({"tier": "filed-cell"}, id="a checked tier nobody checked"),
        pytest.param(
            {"verified_by": "vedant", "verified_at": "t"}, id="an -ai tier marked checked"
        ),
        pytest.param({"tier": "filed-cell", "verified_by": "vedant"}, id="checked with no time"),
        pytest.param({"tier": "xbrl"}, id="a person's xbrl tier with nobody named"),
        pytest.param({"tier": "xbrl-filed"}, id="a document's agreement with no document named"),
        pytest.param(
            {"tier": "xbrl-ai", "verified_by": "sec-6k-html:x.htm", "verified_at": "t"},
            id="an unseen xbrl fact marked checked",
        ),
        pytest.param(
            {"tier": "xbrl-filed", "verified_by": "sec-6k-html:x.htm"}, id="xbrl checked, no time"
        ),
        pytest.param({"tier": "xbrl-filed", "verified_at": "t"}, id="a time and no checker"),
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
            {"tier": "filed-read", "verified_by": "vedant", "verified_at": "t"}, id="a checked read"
        ),
        pytest.param({"period_kind": "instant", "period_start": None}, id="a balance"),
        pytest.param({"tier": "xbrl-ai"}, id="an xbrl fact nobody has seen"),
        pytest.param(
            {
                "tier": "xbrl-filed",
                "verified_by": "nse-pdf:results/X-2025-03-31.pdf",
                "verified_at": "t",
            },
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

# An older database's rows, by id: (tier, verified_by) before, then after.
_MIGRATED = [
    (7, "xbrl", None, "xbrl-ai", None),
    (
        8,
        "xbrl",
        "pdf:results/ITC-2025-03-31.pdf",
        "xbrl-filed",
        "nse-pdf:results/ITC-2025-03-31.pdf",
    ),
    (9, "xbrl", "vedant", "xbrl", "vedant"),
    (
        10,
        "xbrl",
        "pdf:results/ITC-2025-03-31.pdf+vedant",
        "xbrl",
        "nse-pdf:results/ITC-2025-03-31.pdf+vedant",
    ),
    (11, "pdf-cell", "vedant", "filed-cell", "vedant"),
    (12, "pdf-read-ai", None, "filed-read-ai", None),
    (13, "xbrl-ai", None, "xbrl-ai", None),
    (
        14,
        "xbrl-pdf",
        "pdf:results/X-2024-03-31.pdf",
        "xbrl-filed",
        "nse-pdf:results/X-2024-03-31.pdf",
    ),
    (
        15,
        "xbrl-pdf",
        "pdf:results/X-2023-12-31.htm",
        "xbrl-filed",
        "sec-6k-html:results/X-2023-12-31.htm",
    ),
    (16, "pdf-cell-ai", None, "filed-cell-ai", None),
    (17, "pdf-read", "vedant", "filed-read", "vedant"),
]


@pytest.fixture(params=[pytest.param(0, id="unversioned"), pytest.param(3, id="version 3")])
def old_db(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """A database from before schema version 4, one row per way an old row can be checked."""
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript(_OLD_FACTS)
    old.execute(f"PRAGMA user_version = {request.param}")
    old.execute(
        "INSERT INTO filings VALUES (1, 'ITC', 2025, '2025-03-31', 'c', 'xbrl', 'u', 's', 't')"
    )
    for fid, tier, by, *_ in _MIGRATED:
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
    ("fid", "after"),
    [
        pytest.param(fid, (tier_after, by_after), id=f"{tier} checked by {by}")
        for fid, tier, by, tier_after, by_after in _MIGRATED
    ],
)
def test_an_older_database_gives_each_row_its_tier_and_stamp(
    old_db: Path, fid: int, after: tuple[str, str | None]
) -> None:
    conn = connect(old_db)
    row = conn.execute("SELECT tier, verified_by FROM facts WHERE id = ?", (fid,)).fetchone()
    assert row == after, row


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
            100.0,
            ("xbrl-filed", "nse-pdf:results/ITC-2025-03-31.pdf", "t"),
            id="an unchanged value",
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
        """UPDATE facts SET tier = 'xbrl-filed', verified_by = 'nse-pdf:results/ITC-2025-03-31.pdf',
                  verified_at = 't'"""
    )
    sync_xbrl_facts(conn, fid, "ITC", "consolidated", [replace(REVENUE, value=value)], WRITER)
    row = conn.execute("SELECT tier, verified_by, verified_at FROM facts").fetchone()
    assert row == kept, row


@pytest.mark.parametrize(
    ("tier", "checker"),
    [
        pytest.param("xbrl-filed", "nse-pdf:results/ITC-2025-03-31.pdf", id="its document agrees"),
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


GROWTH = DerivedRow(
    "ITC",
    "consolidated",
    "DerivedGrowth",
    "Revenue growth, FY25",
    "2024-04-01",
    "2025-03-31",
    None,
    0.1,
    "pure",
    '{"formula_id": 1}',
)


def _checked(conn: sqlite3.Connection, tier: str) -> list[int]:
    """The ids of the two synced facts, set to `tier` (a checked one names a checker)."""
    sync_xbrl_facts(conn, _filing(conn), "ITC", "consolidated", [REVENUE, ASSETS], WRITER)
    by = None if tier.endswith("-ai") else "nse-pdf:x"
    conn.execute("UPDATE facts SET tier = ?, verified_by = ?, verified_at = ?", (tier, by, by))
    return [r[0] for r in conn.execute("SELECT id FROM facts ORDER BY id")]


def test_write_formula_writes_a_new_formula_and_reuses_a_held_one(
    conn: sqlite3.Connection,
) -> None:
    first = write_formula_proposal(conn, "growth", "(a - b) / b", ["a", "b"], "How much?", WRITER)
    again = write_formula_proposal(conn, "growth", "(a - b) / b", ["a", "b"], "Again?", WRITER)
    other = write_formula_proposal(conn, "growth", "a / b - 1", ["a", "b"], None, WRITER)
    assert first[1] and not again[1] and other[1], (first, again, other)
    assert first[0] == again[0] != other[0]
    assert conn.execute(
        "SELECT name, expression, variables, tier, question, written_by, run_id "
        "FROM formula_proposals WHERE id = ?",
        (first[0],),
    ).fetchone() == (
        "growth",
        "(a - b) / b",
        "a, b",
        "formula-ai",
        "How much?",
        WRITER.written_by,
        "build-1",
    )


@pytest.mark.parametrize(
    ("tier", "by", "at"),
    [
        pytest.param("formula", None, None, id="a checked formula with no checker"),
        pytest.param("formula-ai", "vedant", "t", id="an agent's formula marked checked"),
        pytest.param("formula", "vedant", None, id="a checker with no time"),
        pytest.param("equation", None, None, id="an unknown tier"),
    ],
)
def test_the_formulas_table_refuses_a_row_that_breaks_a_rule(
    conn: sqlite3.Connection, tier: str, by: str | None, at: str | None
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO formula_proposals (name, expression, variables, tier, written_by,
                      written_at, run_id, verified_by, verified_at)
               VALUES ('g', 'a', 'a', ?, 'w', 't', 'r', ?, ?)""",
            (tier, by, at),
        )


def test_write_derived_writes_a_derived_ai_fact_linked_to_its_inputs(
    conn: sqlite3.Connection,
) -> None:
    ids = _checked(conn, "xbrl-filed")
    fact_id = write_derived(conn, GROWTH, [ids[0], ids[1], ids[0]], WRITER)
    assert conn.execute(
        """SELECT ticker, basis, concept, label, period_kind, period_start, period_end, value,
                  unit, tier, formula, written_by, run_id, verified_by FROM facts WHERE id = ?""",
        (fact_id,),
    ).fetchone() == (
        "ITC",
        "consolidated",
        "DerivedGrowth",
        "Revenue growth, FY25",
        "duration",
        "2024-04-01",
        "2025-03-31",
        0.1,
        "pure",
        "derived-ai",
        '{"formula_id": 1}',
        WRITER.written_by,
        "build-1",
        None,
    )
    linked = conn.execute(
        "SELECT input_fact_id FROM derived_from WHERE fact_id = ? ORDER BY input_fact_id",
        (fact_id,),
    ).fetchall()
    assert linked == [(ids[0],), (ids[1],)], "each input once"


def test_write_derived_writes_an_instant_without_a_start(conn: sqlite3.Connection) -> None:
    ids = _checked(conn, "xbrl")
    at = replace(GROWTH, period_start=None)
    fact_id = write_derived(conn, at, ids[:1], WRITER)
    kind = conn.execute("SELECT period_kind FROM facts WHERE id = ?", (fact_id,)).fetchone()
    assert kind == ("instant",)


@pytest.mark.parametrize(
    ("tier", "inputs"),
    [
        pytest.param("xbrl-ai", [0], id="an unchecked input"),
        pytest.param("xbrl-filed", [999], id="a missing input"),
        pytest.param("xbrl-filed", [], id="no input"),
    ],
)
def test_write_derived_refuses_and_writes_nothing(
    conn: sqlite3.Connection, tier: str, inputs: list[int]
) -> None:
    ids = _checked(conn, tier)
    given = [ids[i] if i < len(ids) else i for i in inputs]
    with pytest.raises(ValueError):
        write_derived(conn, GROWTH, given, WRITER)
    assert conn.execute("SELECT count(*) FROM facts WHERE tier = 'derived-ai'").fetchone() == (0,)
    assert conn.execute("SELECT count(*) FROM derived_from").fetchone() == (0,)


@pytest.mark.parametrize(
    ("loaded", "dropped"),
    [
        pytest.param([REVENUE, ASSETS], 0, id="an unchanged filing keeps the derived fact"),
        pytest.param([replace(REVENUE, value=101.0), ASSETS], 1, id="a revised input drops it"),
        pytest.param([replace(REVENUE, decimals=-7), ASSETS], 1, id="an input re-filed drops it"),
        pytest.param([ASSETS], 1, id="a removed input drops it, where it once failed"),
        pytest.param(
            [REVENUE, replace(ASSETS, value=51.0)], 0, id="another fact's change keeps it"
        ),
    ],
)
def test_a_rebuild_drops_derived_facts_whose_input_changed(
    conn: sqlite3.Connection, loaded: list[Fact], dropped: int
) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    revenue, _ = _checked(conn, "xbrl-filed")
    formula_id, _ = write_formula_proposal(conn, "growth", "a / 2", ["a"], None, WRITER)
    derived = write_derived(conn, GROWTH, [revenue], WRITER)

    synced = sync_xbrl_facts(conn, _filing(conn), "ITC", "consolidated", loaded, WRITER)

    assert synced.derived_dropped == dropped, synced
    held = conn.execute("SELECT count(*) FROM facts WHERE id = ?", (derived,)).fetchone()
    links = conn.execute("SELECT count(*) FROM derived_from WHERE fact_id = ?", (derived,))
    assert (held, links.fetchone()) == ((1 - dropped,), (1 - dropped,))
    assert conn.execute("SELECT id FROM formula_proposals").fetchall() == [(formula_id,)], "kept"


def test_write_derived_concept_holds_the_first_by_name_and_scope(
    conn: sqlite3.Connection,
) -> None:
    assert write_derived_concept(
        conn, "NetMargin", "banking", "net over income", "ratio", "{}", "Why?", WRITER
    )
    assert not write_derived_concept(
        conn, "NetMargin", "banking", "another", "growth", "{}", None, WRITER
    )
    assert write_derived_concept(conn, "NetMargin", "ind-as", "other", "ratio", "{}", None, WRITER)
    assert conn.execute(
        """SELECT name, scope, label, formula, tier, question, written_by, run_id
           FROM derived_concepts ORDER BY scope"""
    ).fetchall() == [
        (
            "NetMargin",
            "banking",
            "net over income",
            "ratio",
            "derived-concept-ai",
            "Why?",
            WRITER.written_by,
            "build-1",
        ),
        (
            "NetMargin",
            "ind-as",
            "other",
            "ratio",
            "derived-concept-ai",
            None,
            WRITER.written_by,
            "build-1",
        ),
    ]


@pytest.mark.parametrize(
    ("tier", "by", "at"),
    [
        pytest.param("derived-concept", None, None, id="a checked metric with no checker"),
        pytest.param("derived-concept-ai", "vedant", "t", id="an agent's metric marked checked"),
        pytest.param("derived-concept", "vedant", None, id="a checker with no time"),
        pytest.param("measure", None, None, id="an unknown tier"),
    ],
)
def test_the_derived_concepts_table_refuses_a_row_that_breaks_a_rule(
    conn: sqlite3.Connection, tier: str, by: str | None, at: str | None
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO derived_concepts (name, label, formula, bindings, tier, written_by,
                      written_at, run_id, verified_by, verified_at)
               VALUES ('M', 'm', 'ratio', '{}', ?, 'w', 't', 'r', ?, ?)""",
            (tier, by, at),
        )


def test_an_older_metrics_table_becomes_derived_concepts(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    old = connect(path)
    with old:
        old.execute("DROP TABLE derived_concepts")
        old.execute(
            """CREATE TABLE metrics (name TEXT PRIMARY KEY, label TEXT NOT NULL,
                  formula TEXT NOT NULL, bindings TEXT NOT NULL, tier TEXT NOT NULL,
                  question TEXT, written_by TEXT NOT NULL, written_at TEXT NOT NULL,
                  run_id TEXT NOT NULL, verified_by TEXT, verified_at TEXT)"""
        )
        old.executemany(
            "INSERT INTO metrics VALUES (?, 'l', 'ratio', '{}', ?, NULL, 'w', 't', 'r', ?, ?)",
            [("Unchecked", "metric-ai", None, None), ("Checked", "metric", "vedant", "t")],
        )
    old.close()

    conn = connect(path)

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "metrics" not in tables and "derived_concepts" in tables
    assert conn.execute(
        "SELECT name, scope, tier, verified_by FROM derived_concepts ORDER BY name"
    ).fetchall() == [
        ("Checked", "banking", "derived-concept", "vedant"),
        ("Unchecked", "banking", "derived-concept-ai", None),
    ], "every concept from before scopes came from a bank question"
    conn.close()
    assert connect(path).execute("SELECT count(*) FROM derived_concepts").fetchone() == (2,)


def test_an_older_formulas_table_becomes_formula_proposals(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    old = connect(path)
    with old:
        old.execute("DROP TABLE formula_proposals")
        old.execute(
            """CREATE TABLE formulas (id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                  expression TEXT NOT NULL, variables TEXT NOT NULL, tier TEXT NOT NULL,
                  question TEXT, written_by TEXT NOT NULL, written_at TEXT NOT NULL,
                  run_id TEXT NOT NULL, verified_by TEXT, verified_at TEXT)"""
        )
        old.execute(
            "INSERT INTO formulas VALUES (7, 'cagr', 'x', 'x', 'formula-ai', 'q', 'w', 't', 'r', "
            "NULL, NULL)"
        )
    old.close()
    conn = connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "formulas" not in tables
    assert conn.execute("SELECT id, name, tier FROM formula_proposals").fetchall() == [
        (7, "cagr", "formula-ai")
    ], "its rows, ids and tiers kept"


def test_verify_derived_concept_records_a_person_s_check_once(conn: sqlite3.Connection) -> None:
    write_derived_concept(
        conn, "NetMargin", "banking", "net over income", "ratio", "{}", None, WRITER
    )
    assert verify_derived_concept(
        conn, "NetMargin", "banking", " vedant ", "2026-11-03T23:00:00+05:30"
    ) == (
        True,
        0,
    )
    assert verify_derived_concept(conn, "NetMargin", "banking", "someone", "later") == (False, 0)
    assert conn.execute(
        "SELECT tier, verified_by, verified_at FROM derived_concepts"
    ).fetchone() == ("derived-concept", "vedant", "2026-11-03T23:00:00+05:30")


@pytest.mark.parametrize(
    ("name", "person", "said"),
    [
        pytest.param("Nope", "vedant", "no derived concept", id="a name not held"),
        pytest.param("NetMargin", "  ", "who checked", id="nobody named"),
    ],
)
def test_verify_derived_concept_refuses(
    conn: sqlite3.Connection, name: str, person: str, said: str
) -> None:
    write_derived_concept(
        conn, "NetMargin", "banking", "net over income", "ratio", "{}", None, WRITER
    )
    with pytest.raises(ValueError, match=said):
        verify_derived_concept(conn, name, "banking", person, "t")
    assert conn.execute("SELECT tier FROM derived_concepts").fetchone() == ("derived-concept-ai",)


def test_a_figure_written_with_a_concept_check_is_derived(conn: sqlite3.Connection) -> None:
    ids = _checked(conn, "xbrl-filed")
    fact_id = write_derived(conn, GROWTH, ids[:1], WRITER, checked_by="DerivedGrowth@vedant")
    assert conn.execute(
        "SELECT tier, verified_by, verified_at FROM facts WHERE id = ?", (fact_id,)
    ).fetchone() == ("derived", "DerivedGrowth@vedant", WRITER.written_at)


@pytest.mark.parametrize(
    ("input_tier", "promoted", "tier"),
    [
        pytest.param("xbrl-filed", 1, "derived", id="inputs checked: the figure is trusted"),
        pytest.param("xbrl", 1, "derived", id="inputs checked by a person"),
        pytest.param("xbrl-ai", 0, "derived-ai", id="an input lost its check: it stays unchecked"),
    ],
)
def test_checking_a_concept_promotes_the_figures_computed_with_it(
    conn: sqlite3.Connection, input_tier: str, promoted: int, tier: str
) -> None:
    ids = _checked(conn, "xbrl-filed")
    write_derived_concept(conn, "DerivedGrowth", "ind-as", "growth", "growth", "{}", None, WRITER)
    fact_id = write_derived(conn, GROWTH, ids, WRITER)
    other = write_derived(conn, replace(GROWTH, concept="OtherGrowth"), ids, WRITER)
    by = None if input_tier.endswith("-ai") else "x"
    conn.execute(
        "UPDATE facts SET tier = ?, verified_by = ?, verified_at = ? WHERE id = ?",
        (input_tier, by, by, ids[1]),
    )
    assert verify_derived_concept(conn, "DerivedGrowth", "ind-as", "vedant", "t2") == (
        True,
        promoted,
    )
    row = conn.execute("SELECT tier, verified_by FROM facts WHERE id = ?", (fact_id,)).fetchone()
    assert row == (tier, "DerivedGrowth[ind-as]@vedant" if promoted else None)
    untouched = conn.execute("SELECT tier FROM facts WHERE id = ?", (other,)).fetchone()
    assert untouched == ("derived-ai",), "another concept's figure"


def test_a_derived_concepts_table_from_before_scopes_gets_its_scope(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    old = connect(path)
    with old:
        old.execute("DROP TABLE derived_concepts")
        old.execute(
            """CREATE TABLE derived_concepts (name TEXT PRIMARY KEY, label TEXT NOT NULL,
                  formula TEXT NOT NULL, bindings TEXT NOT NULL, tier TEXT NOT NULL,
                  question TEXT, written_by TEXT NOT NULL, written_at TEXT NOT NULL,
                  run_id TEXT NOT NULL, verified_by TEXT, verified_at TEXT)"""
        )
        old.execute(
            "INSERT INTO derived_concepts VALUES ('NetMargin', 'l', 'ratio', '{}', "
            "'derived-concept', NULL, 'w', 't', 'r', 'vedant', 't')"
        )
    old.close()
    conn = connect(path)
    assert conn.execute(
        "SELECT name, scope, tier, verified_by FROM derived_concepts"
    ).fetchall() == [("NetMargin", "banking", "derived-concept", "vedant")]
    with conn:
        assert write_derived_concept(conn, "NetMargin", "ind-as", "l", "ratio", "{}", None, WRITER)
    conn.close()
    assert connect(path).execute("SELECT count(*) FROM derived_concepts").fetchone() == (2,)


def test_a_check_promotes_only_figures_of_its_own_scope(conn: sqlite3.Connection) -> None:
    ids = _checked(conn, "xbrl-filed")
    write_derived_concept(conn, "DerivedGrowth", "banking", "g", "growth", "{}", None, WRITER)
    itc = write_derived(conn, GROWTH, ids, WRITER)
    assert verify_derived_concept(conn, "DerivedGrowth", "banking", "vedant", "t2") == (True, 0)
    tier = conn.execute("SELECT tier FROM facts WHERE id = ?", (itc,)).fetchone()
    assert tier == ("derived-ai",), "ITC files under ind-as; a banking check says nothing of it"


@pytest.mark.parametrize(
    ("spec", "named", "normal", "refused"),
    [
        pytest.param("banking", ("banking",), "banking", False, id="one scope"),
        pytest.param("ind-as, banking", ("banking", "ind-as"), "*", False, id="both, as all"),
        pytest.param("*", ("banking", "ind-as"), "*", False, id="every scope"),
        pytest.param("bank", ("bank",), "bank", True, id="a part of a name is no scope"),
        pytest.param("banking,*", ("*", "banking"), "*,banking", True, id="star in a list"),
        pytest.param(" , ", (), "", True, id="nothing named"),
    ],
)
def test_a_scope_spec(spec: str, named: tuple[str, ...], normal: str, refused: bool) -> None:
    assert scopes_of(spec) == named
    assert (scope_refusal(spec) is not None) is refused, scope_refusal(spec)
    if not refused:
        assert normal_scope(spec) == normal
        assert all(covers(spec, s) for s in named) and not covers(spec, "insurance")


@pytest.mark.parametrize(
    ("held", "asked", "written"),
    [
        pytest.param("banking", "ind-as", True, id="another scope"),
        pytest.param("banking", "*", False, id="every scope overlaps one held"),
        pytest.param("*", "banking", False, id="one scope inside every scope held"),
        pytest.param("banking", "banking,ind-as", False, id="a list overlapping"),
    ],
)
def test_a_name_is_held_once_in_any_scope(
    conn: sqlite3.Connection, held: str, asked: str, written: bool
) -> None:
    assert write_derived_concept(conn, "M", held, "m", "ratio", "{}", None, WRITER)
    assert write_derived_concept(conn, "M", asked, "m", "ratio", "{}", None, WRITER) is written


def test_a_table_checking_scope_against_a_fixed_list_is_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    old = connect(path)
    with old:
        old.execute("DROP TABLE derived_concepts")
        old.execute(
            """CREATE TABLE derived_concepts (name TEXT NOT NULL,
                  scope TEXT NOT NULL CHECK (scope IN ('banking', 'ind-as')),
                  label TEXT NOT NULL, formula TEXT NOT NULL, bindings TEXT NOT NULL,
                  tier TEXT NOT NULL, question TEXT, written_by TEXT NOT NULL,
                  written_at TEXT NOT NULL, run_id TEXT NOT NULL, verified_by TEXT,
                  verified_at TEXT, PRIMARY KEY (name, scope))"""
        )
        old.execute(
            "INSERT INTO derived_concepts VALUES ('M', 'ind-as', 'l', 'ratio', '{}', "
            "'derived-concept-ai', NULL, 'w', 't', 'r', NULL, NULL)"
        )
    old.close()
    conn = connect(path)
    assert conn.execute("SELECT name, scope FROM derived_concepts").fetchall() == [("M", "ind-as")]
    with conn:
        assert write_derived_concept(conn, "N", "*", "n", "ratio", "{}", None, WRITER)


def test_a_check_on_an_every_scope_concept_promotes_in_both(conn: sqlite3.Connection) -> None:
    ids = _checked(conn, "xbrl-filed")
    write_derived_concept(conn, "DerivedGrowth", "*", "g", "growth", "{}", None, WRITER)
    itc = write_derived(conn, GROWTH, ids, WRITER)
    assert verify_derived_concept(conn, "DerivedGrowth", "banking,ind-as", "vedant", "t") == (
        True,
        1,
    )
    row = conn.execute("SELECT tier, verified_by FROM facts WHERE id = ?", (itc,)).fetchone()
    assert row == ("derived", "DerivedGrowth[*]@vedant")


def _offset_rows(path: Path, *rows: tuple[str, str, str, str]) -> None:
    """A database whose derived concepts are bound by offset: (name, label, bindings, tier)."""
    old = connect(path)
    with old:
        old.executemany(
            """INSERT INTO derived_concepts (name, scope, label, formula, bindings, tier,
                      written_by, written_at, run_id, verified_by, verified_at)
               VALUES (?, 'banking', ?, 'growth', ?, ?, 'w', 't', 'r',
                       CASE ?4 WHEN 'derived-concept' THEN 'vedant' END,
                       CASE ?4 WHEN 'derived-concept' THEN 't' END)""",
            rows,
        )
    old.close()


def _growth(offset: str) -> str:
    return (
        '{"current": {"concept": "Profit", "offset": "t"}, '
        f'"previous": {{"concept": "Profit", "offset": "{offset}"}}}}'
    )


def test_bindings_by_offset_name_the_period_and_a_twin_left_the_same_goes(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    _offset_rows(
        path,
        ("ProfitGrowth", "Growth in profit, year on year", _growth("t-1y"), "derived-concept-ai"),
        (
            "ProfitQuarterOnQuarterGrowth",
            "Growth in profit over the quarter before",
            _growth("t-1q"),
            "derived-concept",
        ),
    )
    conn = connect(path)
    assert conn.execute("SELECT name, label, bindings FROM derived_concepts").fetchall() == [
        (
            "ProfitQuarterOnQuarterGrowth",
            "Growth in profit",
            '{"current": {"at": "current", "concept": "Profit"}, '
            '"previous": {"at": "previous", "concept": "Profit"}}',
        )
    ], "the checked twin is kept, though its name is longer"
    conn.close()
    assert connect(path).execute("SELECT count(*) FROM derived_concepts").fetchone() == (1,)


@pytest.mark.parametrize(
    ("label", "kept"),
    [
        ("Growth in advances, year on year", "Growth in advances"),
        ("Year-on-year growth in operating expenses", "Growth in operating expenses"),
        ("Growth in net profit over the quarter before", "Growth in net profit"),
        ("Quarter-on-quarter growth in deposits", "Growth in deposits"),
        ("Share of other income in total income", "Share of other income in total income"),
    ],
)
def test_a_label_bound_by_offset_loses_its_period_words(
    tmp_path: Path, label: str, kept: str
) -> None:
    _offset_rows(tmp_path / "old.sqlite", ("M", label, _growth("t-1y"), "derived-concept-ai"))
    found = connect(tmp_path / "old.sqlite").execute("SELECT label FROM derived_concepts")
    assert found.fetchone() == (kept,)
