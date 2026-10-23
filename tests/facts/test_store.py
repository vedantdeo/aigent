"""The facts database's schema guards and the build's sync of a filing's XBRL rows."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from aigent.facts.store import (
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
        "xbrl",
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
