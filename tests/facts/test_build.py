"""The build end to end over a scripted NSE: fetched once, loaded, checked, and every failure kept
as data while the rest still builds."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

from aigent.facts.build import Report, build
from aigent.facts.nse import read_manifest, write_manifest
from aigent.facts.store import Writer, connect

from .conftest import INTEGRATED_Q3, LEGACY_Q4, FakeNse, fundamental, integrated_row, legacy_row

NOW = datetime(2026, 10, 22, 12, 0, 0)
WRITER = Writer("xbrl-loader@abc1234", "2026-10-22T12:00:00+05:30", "build-1")
MAR = "https://x/itc-2024-03-31-c.xml"
DEC = "https://x/itc-2025-12-31-c.xml"
ITC_FY24_REVENUE = fundamental("ITC", 2024, "revenue", 768404900000.0)


@pytest.fixture
def scripted(nse: FakeNse) -> FakeNse:
    nse.listings(
        "ITC",
        legacy=[legacy_row("31-Mar-2024", "Consolidated", MAR, "23-May-2024 05:03:07")],
        integrated=[[integrated_row("31-DEC-2025", "Consolidated", DEC, "29-Jan-2026 06:52:14")]],
    )
    nse.answers |= {MAR: LEGACY_Q4.read_bytes(), DEC: INTEGRATED_Q3.read_bytes()}
    return nse


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return connect(tmp_path / "facts.sqlite")


def _build(
    conn: sqlite3.Connection, xbrl: Path, get: FakeNse | None, tickers: tuple[str, ...] = ("ITC",)
) -> Report:
    return build(conn, xbrl, get, tickers, (2024, 2026), WRITER, NOW, [ITC_FY24_REVENUE])


def test_a_build_fetches_loads_and_checks(
    scripted: FakeNse, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    report = _build(conn, tmp_path / "xbrl", scripted)
    assert report.fetched == ["ITC-2024-03-31-C.xml", "ITC-2025-12-31-C.xml"]
    assert (report.fetch_failures, report.load_failures) == ([], [])
    assert [(x.name, x.issues, x.sum_failures) for x in report.loaded] == [
        ("ITC-2024-03-31-C.xml", [], []),
        ("ITC-2025-12-31-C.xml", [], []),
    ]
    assert [c.status for c in report.fundamentals] == ["equal"]
    filings = conn.execute("SELECT ticker, fy, period_end, basis, origin FROM filings").fetchall()
    assert sorted(filings) == [
        ("ITC", 2024, "2024-03-31", "consolidated", MAR),
        ("ITC", 2026, "2025-12-31", "consolidated", DEC),
    ]
    stored = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    assert stored == sum(x.facts for x in report.loaded) > 0
    assert "fundamentals 1 equal" in report.summary()
    json.dumps(report.as_json(), default=str)


def test_a_second_build_fetches_nothing_and_changes_nothing(
    scripted: FakeNse, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    _build(conn, tmp_path / "xbrl", scripted)
    calls = len(scripted.calls)
    again = _build(conn, tmp_path / "xbrl", scripted)
    assert again.fetched == []
    assert [u for u in scripted.calls[calls:] if u in (MAR, DEC)] == []
    assert {(x.synced.added, x.synced.changed, x.synced.removed) for x in again.loaded} == {
        (0, 0, 0)
    }


def test_an_offline_build_loads_the_cache_without_asking_nse(
    scripted: FakeNse, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    _build(conn, tmp_path / "xbrl", scripted)
    calls = len(scripted.calls)
    report = _build(conn, tmp_path / "xbrl", None)
    assert len(scripted.calls) == calls
    assert len(report.loaded) == 2


def test_a_company_whose_listing_fails_is_reported_and_the_rest_build(
    scripted: FakeNse, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    report = _build(conn, tmp_path / "xbrl", scripted, ("WIPRO", "ITC"))
    assert len(report.fetch_failures) == 1 and report.fetch_failures[0].startswith("WIPRO")
    assert len(report.loaded) == 2


def test_a_file_that_fails_to_download_is_reported_and_the_rest_build(
    scripted: FakeNse, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    scripted.answers[DEC] = OSError("HTTP Error 403")
    report = _build(conn, tmp_path / "xbrl", scripted)
    assert report.fetch_failures == ["ITC-2025-12-31-C.xml: HTTP Error 403"]
    assert [x.name for x in report.loaded] == ["ITC-2024-03-31-C.xml"]


@pytest.mark.parametrize(
    ("spoil", "failure"),
    [
        pytest.param(
            lambda p: p.write_bytes(b"<changed/>"),
            "is not the manifest's",
            id="a file changed since it was fetched",
        ),
        pytest.param(lambda p: p.unlink(), "not on disk", id="a file gone from the cache"),
    ],
)
def test_a_cached_file_that_cannot_be_trusted_is_not_loaded(
    scripted: FakeNse,
    conn: sqlite3.Connection,
    tmp_path: Path,
    spoil: Callable[[Path], object],
    failure: str,
) -> None:
    _build(conn, tmp_path / "xbrl", scripted)
    spoil(tmp_path / "xbrl" / "ITC-2025-12-31-C.xml")
    report = _build(conn, tmp_path / "xbrl", None)
    assert [f for f in report.load_failures if failure in f] != [], report.load_failures
    assert [x.name for x in report.loaded] == ["ITC-2024-03-31-C.xml"]


def test_an_unreadable_file_is_reported(
    scripted: FakeNse, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    scripted.answers[DEC] = b"<not xml"
    report = _build(conn, tmp_path / "xbrl", scripted)
    assert [f.split(":")[0] for f in report.load_failures] == ["ITC-2025-12-31-C.xml"]


def test_a_cached_file_out_of_scope_is_not_loaded(
    scripted: FakeNse, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    xbrl = tmp_path / "xbrl"
    _build(conn, xbrl, scripted)
    manifest = read_manifest(xbrl)
    manifest["WIPRO-2024-03-31-C.xml"] = manifest["ITC-2024-03-31-C.xml"] | {"ticker": "WIPRO"}
    manifest["ITC-2023-03-31-C.xml"] = manifest["ITC-2024-03-31-C.xml"] | {
        "period_end": "2023-03-31"
    }
    write_manifest(xbrl, manifest)
    for name in ("WIPRO-2024-03-31-C.xml", "ITC-2023-03-31-C.xml"):
        (xbrl / name).write_bytes((xbrl / "ITC-2024-03-31-C.xml").read_bytes())
    report = _build(conn, xbrl, None)
    assert [x.name for x in report.loaded] == ["ITC-2024-03-31-C.xml", "ITC-2025-12-31-C.xml"]
