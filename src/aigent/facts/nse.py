"""Lists and fetches NSE's XBRL results filings: one file per company, quarter and basis, kept once
under the cache with its URL and SHA-256 in a sorted manifest, and never fetched again."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast

from aigent.facts.cfg_facts import (
    NSE_INTEGRATED_LISTING,
    NSE_LEGACY_LISTING,
    NSE_PAGE_SIZE,
    NSE_REQUEST_INTERVAL_S,
    NSE_SYMBOLS,
    NSE_TIMEOUT_S,
    NSE_USER_AGENT,
)

Basis = Literal["consolidated", "standalone"]
Get = Callable[[str], bytes]

XBRL_DIR = Path(__file__).resolve().parents[3] / "corpus" / ".cache" / "xbrl"
MANIFEST_NAME = "manifest.json"

_BASES: dict[str, Basis] = {
    "Consolidated": "consolidated",
    "Non-Consolidated": "standalone",
    "Standalone": "standalone",
}


@dataclass(frozen=True)
class Listed:
    """One results filing as NSE lists it."""

    ticker: str
    basis: Basis
    period_end: date
    url: str
    broadcast: datetime

    @property
    def fy(self) -> int:
        """The financial year the quarter falls in: Dec 2024 is in 2025 (FY 2024-25)."""
        return self.period_end.year + (1 if self.period_end.month > 3 else 0)

    @property
    def name(self) -> str:
        return f"{self.ticker}-{self.period_end.isoformat()}-{self.basis[0].upper()}.xml"


def symbol_of(ticker: str) -> str:
    return NSE_SYMBOLS.get(ticker, ticker)


def polite_get(
    interval: float = NSE_REQUEST_INTERVAL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Get:
    """A getter that waits so requests are at least `interval` seconds apart."""
    last: list[float] = []

    def get(url: str) -> bytes:
        if last:
            wait = last[0] + interval - clock()
            if wait > 0:
                sleep(wait)
        request = urllib.request.Request(
            url, headers={"User-Agent": NSE_USER_AGENT, "Accept": "application/json, */*"}
        )
        try:
            with urllib.request.urlopen(request, timeout=NSE_TIMEOUT_S) as response:
                return cast(bytes, response.read())
        finally:
            last[:] = [clock()]

    return get


def _legacy_rows(ticker: str, rows: Iterable[dict[str, object]]) -> list[Listed]:
    out: list[Listed] = []
    for row in rows:
        basis, url = _BASES.get(str(row.get("consolidated"))), row.get("xbrl")
        if basis is None or not isinstance(url, str) or not url.endswith(".xml"):
            continue
        out.append(
            Listed(
                ticker=ticker,
                basis=basis,
                period_end=datetime.strptime(str(row["toDate"]), "%d-%b-%Y").date(),
                url=url,
                broadcast=datetime.strptime(str(row["broadCastDate"]), "%d-%b-%Y %H:%M:%S"),
            )
        )
    return out


def _integrated_rows(ticker: str, rows: Iterable[dict[str, object]]) -> list[Listed]:
    out: list[Listed] = []
    for row in rows:
        basis, url = _BASES.get(str(row.get("consolidated"))), row.get("xbrl")
        if basis is None or not isinstance(url, str) or not url.endswith(".xml"):
            continue
        out.append(
            Listed(
                ticker=ticker,
                basis=basis,
                period_end=datetime.strptime(str(row["qe_Date"]).title(), "%d-%b-%Y").date(),
                url=url,
                broadcast=datetime.strptime(str(row["broadcast_Date"]), "%d-%b-%Y %H:%M:%S"),
            )
        )
    return out


def list_filings(get: Get, ticker: str, years: Iterable[int]) -> list[Listed]:
    """The ticker's results filings in `years`, the latest broadcast per quarter and basis, so a
    revision replaces what it revises."""
    symbol = symbol_of(ticker)
    legacy = json.loads(get(NSE_LEGACY_LISTING.format(symbol=symbol)))
    listed = _legacy_rows(ticker, cast(list[dict[str, object]], legacy))
    page, seen = 1, 0
    while True:
        body = cast(
            dict[str, object],
            json.loads(
                get(NSE_INTEGRATED_LISTING.format(symbol=symbol, size=NSE_PAGE_SIZE, page=page))
            ),
        )
        rows = cast(list[dict[str, object]], body.get("data") or [])
        listed += _integrated_rows(ticker, rows)
        seen += len(rows)
        if not rows or seen >= int(cast(int, body.get("totalCount") or 0)):
            break
        page += 1
    wanted = set(years)
    latest: dict[tuple[date, Basis], Listed] = {}
    for filing in listed:
        if filing.fy not in wanted:
            continue
        held = latest.get((filing.period_end, filing.basis))
        if held is None or filing.broadcast > held.broadcast:
            latest[(filing.period_end, filing.basis)] = filing
    return [latest[key] for key in sorted(latest)]


def read_manifest(xbrl_dir: Path) -> dict[str, dict[str, str]]:
    path = xbrl_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    return cast(dict[str, dict[str, str]], json.loads(path.read_text(encoding="utf-8")))


def write_manifest(xbrl_dir: Path, manifest: dict[str, dict[str, str]]) -> None:
    """Written sorted by file name, each entry's fields in one order."""
    xbrl_dir.mkdir(parents=True, exist_ok=True)
    ordered = {name: dict(sorted(manifest[name].items())) for name in sorted(manifest)}
    (xbrl_dir / MANIFEST_NAME).write_text(json.dumps(ordered, indent=1) + "\n", encoding="utf-8")


def fetch(
    get: Get, filing: Listed, xbrl_dir: Path, manifest: dict[str, dict[str, str]], now: datetime
) -> bool:
    """Keeps the filing's file unless the manifest already holds it from the same URL; True if it
    was fetched. A different URL under the same name is a revision and replaces it."""
    held = manifest.get(filing.name)
    path = xbrl_dir / filing.name
    if held is not None and held["url"] == filing.url and path.exists():
        return False
    data = get(filing.url)
    xbrl_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    manifest[filing.name] = {
        "basis": filing.basis,
        "broadcast": filing.broadcast.isoformat(),
        "fetched": now.isoformat(timespec="seconds"),
        "period_end": filing.period_end.isoformat(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "ticker": filing.ticker,
        "url": filing.url,
    }
    return True
