"""`python -m aigent.facts build`: fetch what is missing, load every cached filing's XBRL facts, and
check them. Free; no model calls. A failed fetch is reported and the rest still build."""

from __future__ import annotations

import hashlib
import sqlite3
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from aigent.facts.check import Checked, check_fundamentals, check_sums
from aigent.facts.nse import Get, fetch, list_filings, read_manifest, write_manifest
from aigent.facts.reference import Fundamental
from aigent.facts.store import (
    FilingRow,
    Synced,
    Writer,
    apply_labels,
    sync_xbrl_facts,
    upsert_filing,
)
from aigent.facts.taxonomy import fetch_packages, load_labels
from aigent.facts.xbrl import parse


@dataclass
class Loaded:
    """One filing's load: its counts, what could not be read, and the sums it breaks."""

    name: str
    facts: int
    skipped: int
    synced: Synced
    issues: list[str]
    sum_failures: list[str]


@dataclass
class Report:
    fetched: list[str] = field(default_factory=list[str])
    fetch_failures: list[str] = field(default_factory=list[str])
    loaded: list[Loaded] = field(default_factory=list[Loaded])
    load_failures: list[str] = field(default_factory=list[str])
    fundamentals: list[Checked] = field(default_factory=list[Checked])
    taxonomy_fetched: list[str] = field(default_factory=list[str])
    taxonomy_failures: list[str] = field(default_factory=list[str])
    unlabelled: list[str] = field(default_factory=list[str])

    def as_json(self) -> dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        added = sum(x.synced.added for x in self.loaded)
        changed = sum(x.synced.changed for x in self.loaded)
        removed = sum(x.synced.removed for x in self.loaded)
        counts: dict[str, int] = {}
        for item in self.fundamentals:
            counts[item.status] = counts.get(item.status, 0) + 1
        lines = [
            f"fetched {len(self.fetched)} files, {len(self.fetch_failures)} fetch failures",
            f"loaded {len(self.loaded)} filings, {sum(x.facts for x in self.loaded)} facts "
            f"({added} added, {changed} changed, {removed} removed), "
            f"{sum(x.skipped for x in self.loaded)} out of scope, "
            f"{len(self.load_failures)} unreadable",
            f"issues {sum(len(x.issues) for x in self.loaded)}, "
            f"sums broken {sum(len(x.sum_failures) for x in self.loaded)}",
            "fundamentals " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())),
            f"taxonomy {len(self.taxonomy_fetched)} packages fetched, "
            f"{len(self.taxonomy_failures)} failures, {len(self.unlabelled)} concepts unlabelled",
        ]
        return "\n".join(lines)


def fetch_all(
    get: Get,
    xbrl_dir: Path,
    tickers: Iterable[str],
    years: Iterable[int],
    now: datetime,
    report: Report,
) -> None:
    """Every listed filing not yet cached; the manifest is saved after each company."""
    manifest = read_manifest(xbrl_dir)
    wanted = tuple(years)
    for ticker in tickers:
        try:
            listed = list_filings(get, ticker, wanted)
        except (OSError, ValueError, KeyError) as error:
            report.fetch_failures.append(f"{ticker}: listing failed: {error}")
            continue
        for filing in listed:
            try:
                if fetch(get, filing, xbrl_dir, manifest, now):
                    report.fetched.append(filing.name)
            except OSError as error:
                report.fetch_failures.append(f"{filing.name}: {error}")
        write_manifest(xbrl_dir, manifest)


def load_all(
    conn: sqlite3.Connection,
    xbrl_dir: Path,
    tickers: Iterable[str],
    years: Iterable[int],
    writer: Writer,
    report: Report,
) -> None:
    """Every cached filing in scope into the database, in one transaction."""
    wanted_tickers, wanted_years = set(tickers), set(years)
    with conn:
        for name, entry in sorted(read_manifest(xbrl_dir).items()):
            period_end = date.fromisoformat(entry["period_end"])
            fy = period_end.year + (1 if period_end.month > 3 else 0)
            if entry["ticker"] not in wanted_tickers or fy not in wanted_years:
                continue
            path = xbrl_dir / name
            if not path.exists():
                report.load_failures.append(f"{name}: in the manifest, not on disk")
                continue
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                report.load_failures.append(f"{name}: its SHA-256 is not the manifest's")
                continue
            try:
                parsed = parse(data)
            except (ET.ParseError, ValueError) as error:
                report.load_failures.append(f"{name}: unreadable: {error}")
                continue
            filing_id = upsert_filing(
                conn,
                FilingRow(
                    ticker=entry["ticker"],
                    fy=fy,
                    period_end=period_end,
                    basis=entry["basis"],
                    source="xbrl",
                    origin=entry["url"],
                    sha256=entry["sha256"],
                    fetched=entry["fetched"],
                ),
            )
            synced = sync_xbrl_facts(
                conn, filing_id, entry["ticker"], entry["basis"], parsed.facts, writer
            )
            report.loaded.append(
                Loaded(
                    name=name,
                    facts=len(parsed.facts),
                    skipped=parsed.skipped,
                    synced=synced,
                    issues=parsed.issues,
                    sum_failures=check_sums(conn, filing_id),
                )
            )


def label_all(
    conn: sqlite3.Connection,
    taxonomy_dir: Path,
    get: Get | None,
    packages: Iterable[tuple[str, str]],
    now: datetime,
    report: Report,
) -> None:
    """Fetch the taxonomy packages not yet cached (unless `get` is None), then label every
    concept from them. A package that cannot be fetched or read is reported and nothing is
    relabelled, so a partial set never overrides a label."""
    packages = tuple(packages)
    if get is not None:
        try:
            report.taxonomy_fetched = fetch_packages(get, taxonomy_dir, packages, now)
        except OSError as error:
            report.taxonomy_failures.append(f"fetch failed: {error}")
            return
    try:
        labels = load_labels(taxonomy_dir, [name for name, _ in packages])
    except (ValueError, OSError, ET.ParseError, zipfile.BadZipFile) as error:
        report.taxonomy_failures.append(str(error))
        return
    with conn:
        report.unlabelled = apply_labels(conn, labels)


def build(
    conn: sqlite3.Connection,
    xbrl_dir: Path,
    get: Get | None,
    tickers: Iterable[str],
    years: Iterable[int],
    writer: Writer,
    now: datetime,
    fundamentals: Iterable[Fundamental],
    taxonomy_dir: Path,
    packages: Iterable[tuple[str, str]],
) -> Report:
    """Fetch (unless `get` is None), load, label and check."""
    tickers, years = tuple(tickers), tuple(years)
    report = Report()
    if get is not None:
        fetch_all(get, xbrl_dir, tickers, years, now, report)
    load_all(conn, xbrl_dir, tickers, years, writer, report)
    label_all(conn, taxonomy_dir, get, packages, now, report)
    report.fundamentals = check_fundamentals(conn, fundamentals)
    return report
