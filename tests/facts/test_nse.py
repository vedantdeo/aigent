"""NSE's listings and the XBRL cache: which filings are chosen, what is fetched, how politely."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from aigent.facts import nse
from aigent.facts.cfg_facts import NSE_INTEGRATED_LISTING, NSE_LEGACY_LISTING, NSE_PAGE_SIZE
from aigent.facts.nse import Listed, fetch, list_filings, polite_get, read_manifest, write_manifest

from .conftest import FakeNse, integrated_row, legacy_row

NOW = datetime(2026, 10, 22, 12, 0, 0)


def _listed(url: str = "https://x/a.xml") -> Listed:
    return Listed("ITC", "consolidated", date(2024, 12, 31), url, NOW)


def test_list_filings_chooses_each_quarters_latest_filing_in_the_years(nse: FakeNse) -> None:
    nse.listings(
        "ITC",
        legacy=[
            legacy_row(
                "31-Dec-2024", "Consolidated", "https://x/dec-c.xml", "06-Feb-2025 19:20:20"
            ),
            legacy_row(
                "31-Dec-2024", "Consolidated", "https://x/dec-c-rev.xml", "10-Feb-2025 10:00:00"
            ),
            legacy_row(
                "31-Dec-2024", "Non-Consolidated", "https://x/dec-s.xml", "06-Feb-2025 19:15:00"
            ),
            legacy_row("30-Jun-2022", "Consolidated", "https://x/old.xml", "01-Aug-2022 18:15:59"),
            legacy_row("30-Sep-2024", "Consolidated", "-", "24-Oct-2024 07:35:05"),
        ],
        integrated=[
            [
                integrated_row(
                    "31-MAR-2025", "Consolidated", "https://x/mar-c.xml", "22-May-2025 06:18:57"
                ),
                integrated_row(
                    "31-MAR-2025", None, "https://x/governance.xml", "30-Jun-2025 07:08:50"
                ),
            ],
            [
                integrated_row(
                    "31-MAR-2025", "Standalone", "https://x/mar-s.xml", "22-May-2025 06:18:42"
                )
            ],
        ],
    )
    listed = list_filings(nse, "ITC", (2025,))
    assert [(f.name, f.url) for f in listed] == [
        ("ITC-2024-12-31-C.xml", "https://x/dec-c-rev.xml"),
        ("ITC-2024-12-31-S.xml", "https://x/dec-s.xml"),
        ("ITC-2025-03-31-C.xml", "https://x/mar-c.xml"),
        ("ITC-2025-03-31-S.xml", "https://x/mar-s.xml"),
    ]


def test_list_filings_asks_for_the_next_page_until_all_rows_are_read(nse: FakeNse) -> None:
    rows = [
        integrated_row("31-MAR-2025", "Consolidated", "https://x/c.xml", "22-May-2025 06:18:57")
    ]
    nse.listings("ITC", legacy=[], integrated=[rows, rows, rows])
    list_filings(nse, "ITC", (2025,))
    pages = [u for u in nse.calls if "integrated" in u]
    assert pages == [
        NSE_INTEGRATED_LISTING.format(symbol="ITC", size=NSE_PAGE_SIZE, page=n) for n in (1, 2, 3)
    ]


def test_list_filings_asks_nse_by_its_symbol_and_keeps_our_ticker(nse: FakeNse) -> None:
    nse.listings(
        "TMPV",
        legacy=[
            legacy_row("31-Mar-2024", "Consolidated", "https://x/t.xml", "10-May-2024 10:00:00")
        ],
        integrated=[],
    )
    listed = list_filings(nse, "TATAMOTORS", (2024,))
    assert nse.calls[0] == NSE_LEGACY_LISTING.format(symbol="TMPV")
    assert [f.name for f in listed] == ["TATAMOTORS-2024-03-31-C.xml"]


@pytest.mark.parametrize(
    ("period_end", "fy"),
    [
        (date(2024, 6, 30), 2025),
        (date(2024, 12, 31), 2025),
        (date(2025, 3, 31), 2025),
        (date(2025, 4, 30), 2026),
    ],
)
def test_a_quarter_falls_in_the_financial_year_ending_the_next_march(
    period_end: date, fy: int
) -> None:
    assert Listed("ITC", "standalone", period_end, "u", NOW).fy == fy


def test_fetch_keeps_a_file_once_and_never_fetches_it_again(nse: FakeNse, tmp_path: Path) -> None:
    nse.answers["https://x/a.xml"] = b"<xbrl/>"
    manifest: dict[str, dict[str, str]] = {}
    assert fetch(nse, _listed(), tmp_path, manifest, NOW) is True
    assert fetch(nse, _listed(), tmp_path, manifest, NOW) is False
    assert nse.calls == ["https://x/a.xml"]
    entry = manifest["ITC-2024-12-31-C.xml"]
    assert entry["sha256"] == hashlib.sha256(b"<xbrl/>").hexdigest()
    assert (entry["url"], entry["period_end"], entry["basis"]) == (
        "https://x/a.xml",
        "2024-12-31",
        "consolidated",
    )
    assert (tmp_path / "ITC-2024-12-31-C.xml").read_bytes() == b"<xbrl/>"


def test_fetch_replaces_a_file_when_a_revision_moves_its_url(nse: FakeNse, tmp_path: Path) -> None:
    nse.answers |= {"https://x/a.xml": b"<old/>", "https://x/b.xml": b"<new/>"}
    manifest: dict[str, dict[str, str]] = {}
    fetch(nse, _listed("https://x/a.xml"), tmp_path, manifest, NOW)
    assert fetch(nse, _listed("https://x/b.xml"), tmp_path, manifest, NOW) is True
    assert (tmp_path / "ITC-2024-12-31-C.xml").read_bytes() == b"<new/>"


def test_fetch_refetches_a_file_the_manifest_holds_but_the_disk_lost(
    nse: FakeNse, tmp_path: Path
) -> None:
    nse.answers["https://x/a.xml"] = b"<xbrl/>"
    manifest: dict[str, dict[str, str]] = {}
    fetch(nse, _listed(), tmp_path, manifest, NOW)
    (tmp_path / "ITC-2024-12-31-C.xml").unlink()
    assert fetch(nse, _listed(), tmp_path, manifest, NOW) is True


def test_the_manifest_is_written_sorted_by_name_and_field(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"b.xml": {"url": "u", "basis": "s"}, "a.xml": {"z": "1", "a": "2"}})
    raw = json.loads((tmp_path / "manifest.json").read_text())
    assert list(raw) == ["a.xml", "b.xml"]
    assert [list(v) for v in raw.values()] == [["a", "z"], ["basis", "url"]]
    assert read_manifest(tmp_path) == raw


def test_an_absent_manifest_reads_as_empty(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) == {}


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_polite_get_spaces_requests_and_sends_a_browser_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, str | None]] = []

    def urlopen(request: nse.urllib.request.Request, timeout: float) -> _Response:
        sent.append((request.full_url, request.get_header("User-agent")))
        return _Response(b"ok")

    monkeypatch.setattr(nse.urllib.request, "urlopen", urlopen)
    now = [100.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    get = polite_get(1.0, clock=lambda: now[0], sleep=sleep)
    assert get("https://x/1") == b"ok"
    now[0] += 0.25
    get("https://x/2")
    now[0] += 3.0
    get("https://x/3")
    assert slept == [0.75]
    assert [url for url, _ in sent] == ["https://x/1", "https://x/2", "https://x/3"]
    assert all(agent and agent.startswith("Mozilla/") for _, agent in sent)


def test_polite_get_waits_after_a_failed_request_too(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request: nse.urllib.request.Request, timeout: float) -> _Response:
        raise OSError("refused")

    monkeypatch.setattr(nse.urllib.request, "urlopen", urlopen)
    slept: list[float] = []
    get = polite_get(1.0, clock=lambda: 5.0, sleep=slept.append)
    for _ in range(2):
        with pytest.raises(OSError):
            get("https://x/1")
    assert slept == [1.0]
