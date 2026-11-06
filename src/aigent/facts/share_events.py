"""Share-count events per company (bonus issues, splits, issues for value) and per-share figures
restated across them, so EPS filed before a bonus compares with EPS filed after. Events are seeded
from `SHARE_EVENTS` or detected from a jump in the filed share count; each is `share-event-ai`
until a person checks it, and only a checked event restates."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

from aigent.facts.cfg_facts import SHARE_JUMP_MIN
from aigent.facts.store import RESTATING_KINDS, XBRL_TIERS, Writer

SHARE_EVENTS = Path(__file__).resolve().parents[3] / "evals" / "reference" / "share-events.jsonl"
PER_SHARE = "PerShare"  # the unit suffix of a per-share figure, as in INRPerShare
_PAID_UP, _FACE = "PaidUpValueOfEquityShareCapital", "FaceValueOfEquityShareCapital"
_COLUMNS = """id, ticker, kind, effective, shares_after, shares_before, seen_from, seen_to,
    seen_ratio, note, tier, verified_by"""
VERIFY_HOW = "python -m aigent.facts verify-share-event --id N --by NAME"


@dataclass(frozen=True)
class ShareEvent:
    """One held event: a dated bonus, split or issue, or an unknown change seen between two
    filings' period ends."""

    id: int
    ticker: str
    kind: str
    effective: str | None
    shares_after: int | None
    shares_before: int | None
    seen_from: str | None
    seen_to: str | None
    seen_ratio: float | None
    note: str | None
    tier: str
    verified_by: str | None

    @property
    def checked(self) -> bool:
        return self.tier == "share-event"

    @property
    def multiplier(self) -> float:
        """What a per-share figure filed before it is multiplied by: 1/2 for a 1:1 bonus."""
        if self.shares_after is None or self.shares_before is None:
            return 1.0
        return self.shares_before / self.shares_after

    def described(self) -> str:
        if self.kind == "unknown":
            seen = f" (x{self.seen_ratio:.3f})" if self.seen_ratio is not None else ""
            return (
                f"{self.ticker} event {self.id}: an unknown share-count change between "
                f"{self.seen_from} and {self.seen_to}{seen}, {self.tier}"
            )
        ratio = f" {self.shares_after}:{self.shares_before}" if self.shares_after else ""
        return f"{self.ticker} event {self.id}: {self.kind}{ratio} effective {self.effective}, " + (
            f"checked by {self.verified_by}" if self.checked else self.tier
        )


@dataclass(frozen=True)
class Restated:
    """A per-share input's multiplier to the latest filing's share basis, and its events."""

    multiplier: float
    events: tuple[str, ...]


def load_share_events(conn: sqlite3.Connection, ticker: str | None = None) -> list[ShareEvent]:
    """Every event held, or one company's, by company and date."""
    rows = conn.execute(
        f"""SELECT {_COLUMNS} FROM share_events WHERE ? IS NULL OR ticker = ?
            ORDER BY ticker, COALESCE(effective, seen_to), id""",
        (ticker, ticker),
    )
    return [ShareEvent(*row) for row in rows]


def share_counts(conn: sqlite3.Connection, ticker: str) -> list[tuple[str, float]]:
    """Each standalone filing's share count at its period end, paid-up capital over face value, by
    period end."""
    marks = ", ".join("?" for _ in XBRL_TIERS)
    rows = conn.execute(
        f"""SELECT g.period_end, MAX(p.value / f.value) FROM filings g
            JOIN facts p ON p.filing_id = g.id AND p.concept = ? AND p.period_end = g.period_end
                AND p.segment IS NULL AND p.tier IN ({marks})
            JOIN facts f ON f.filing_id = g.id AND f.concept = ? AND f.period_end = g.period_end
                AND f.segment IS NULL AND f.tier IN ({marks}) AND f.value > 0
            WHERE g.ticker = ? AND g.basis = 'standalone'
            GROUP BY g.period_end ORDER BY g.period_end""",
        (_PAID_UP, *XBRL_TIERS, _FACE, *XBRL_TIERS, ticker),
    )
    return [(str(end), float(count)) for end, count in rows]


def _stamp(writer: Writer) -> tuple[str, str, str]:
    return writer.written_by, writer.written_at, writer.run_id


def seed_share_events(conn: sqlite3.Connection, writer: Writer, path: Path = SHARE_EVENTS) -> int:
    """Writes each event in `path` not yet held for its company and date, as `share-event-ai`;
    how many. A row the schema refuses raises."""
    written = 0
    with conn:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = cast(dict[str, object], json.loads(line))
            held = conn.execute(
                "SELECT 1 FROM share_events WHERE ticker = ? AND effective = ?",
                (row["ticker"], row["effective"]),
            ).fetchone()
            if held is not None:
                continue
            conn.execute(
                """INSERT INTO share_events (ticker, kind, effective, shares_after,
                          shares_before, note, tier, written_by, written_at, run_id)
                   VALUES (?, ?, ?, ?, ?, ?, 'share-event-ai', ?, ?, ?)""",
                (
                    row["ticker"],
                    row["kind"],
                    row["effective"],
                    row.get("shares_after"),
                    row.get("shares_before"),
                    row.get("note"),
                    *_stamp(writer),
                ),
            )
            written += 1
    return written


def _within(event: ShareEvent, start: str, end: str) -> bool:
    """Whether `event` may have taken effect after `start` and by `end`, ISO dates."""
    if event.effective is not None:
        return start < event.effective <= end
    return str(event.seen_from) < end and str(event.seen_to) > start


def detect_share_events(conn: sqlite3.Connection, writer: Writer) -> list[ShareEvent]:
    """Holds an unknown event for each move of more than `SHARE_JUMP_MIN` in a company's filed
    share count between two filings with no event held between them; the new ones."""
    made: list[int] = []
    with conn:
        for (ticker,) in conn.execute("SELECT DISTINCT ticker FROM filings ORDER BY ticker"):
            held = load_share_events(conn, str(ticker))
            counts = share_counts(conn, str(ticker))
            for (was_end, was), (end, now) in zip(counts, counts[1:], strict=False):
                ratio = now / was
                if abs(ratio - 1) <= SHARE_JUMP_MIN or any(_within(e, was_end, end) for e in held):
                    continue
                cursor = conn.execute(
                    """INSERT INTO share_events (ticker, kind, seen_from, seen_to, seen_ratio,
                              tier, written_by, written_at, run_id)
                       VALUES (?, 'unknown', ?, ?, ?, 'share-event-ai', ?, ?, ?)""",
                    (ticker, was_end, end, ratio, *_stamp(writer)),
                )
                made.append(cast(int, cursor.lastrowid))
    return [e for e in load_share_events(conn) if e.id in made]


def sync_share_events(
    conn: sqlite3.Connection, writer: Writer, path: Path = SHARE_EVENTS
) -> tuple[int, list[ShareEvent]]:
    """Seeds the events in `path`, then detects the unknown ones; how many were seeded, and the
    unknown events new this time."""
    seeder = Writer(f"seed|{path.name}", writer.written_at, writer.run_id)
    return seed_share_events(conn, seeder, path), detect_share_events(conn, writer)


def _ratio(text: str) -> tuple[int, int]:
    """Shares after and before from `after:before`, 2:1 for a 1:1 bonus."""
    parts = text.split(":")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"ratio is shares after:before, such as 2:1 for a 1:1 bonus, not {text!r}")
    after, before = int(parts[0]), int(parts[1])
    if after <= 0 or before <= 0 or after == before:
        raise ValueError(f"ratio {text!r} changes no share count; after and before differ")
    return after, before


def verify_share_event(
    conn: sqlite3.Connection,
    event_id: int,
    person: str,
    at: str,
    kind: str | None = None,
    effective: str | None = None,
    ratio: str | None = None,
) -> tuple[bool, ShareEvent]:
    """Records `person`'s check of an event, making it `share-event`. For an unknown one they name
    its kind (bonus, split or issue) and date, and for a bonus or split its ratio; given on any
    event, these replace what it held. Returns whether anything changed, and the event. ValueError
    for an event not held, no person, or details that cannot describe an event."""
    found = [e for e in load_share_events(conn) if e.id == event_id]
    if not found:
        raise ValueError(f"no share event {event_id}")
    event = found[0]
    if not person.strip():
        raise ValueError("say who checked it")
    kind = kind or (None if event.kind == "unknown" else event.kind)
    if kind is None:
        raise ValueError("an unknown event needs its kind: bonus, split or issue")
    if kind not in (*RESTATING_KINDS, "issue"):
        raise ValueError(f"kind is bonus, split or issue, not {kind!r}")
    effective = effective or event.effective
    if effective is None:
        raise ValueError("an unknown event needs the date its shares changed, such as 2025-08-28")
    try:
        date.fromisoformat(effective)
    except ValueError:
        raise ValueError(f"{effective!r} is not a date such as 2025-08-28") from None
    if event.seen_from is not None and not str(event.seen_from) < effective <= str(event.seen_to):
        raise ValueError(
            f"the share count changed after {event.seen_from} and by {event.seen_to}; "
            f"{effective} is outside that"
        )
    if kind not in RESTATING_KINDS:
        if ratio:
            raise ValueError("an issue for value restates nothing; give no ratio")
        after, before = None, None
    elif ratio:
        after, before = _ratio(ratio)
    elif event.shares_after is not None:
        after, before = event.shares_after, event.shares_before
    else:
        raise ValueError("a bonus or split needs its ratio, shares after:before such as 2:1")
    given = (kind, effective, after, before)
    held = (event.kind, event.effective, event.shares_after, event.shares_before)
    if event.checked and given == held:
        return False, event
    try:
        with conn:
            conn.execute(
                """UPDATE share_events SET kind = ?, effective = ?, shares_after = ?,
                          shares_before = ?, tier = 'share-event', verified_by = ?,
                          verified_at = ?
                   WHERE id = ?""",
                (*given, person.strip(), at, event_id),
            )
    except sqlite3.IntegrityError:
        raise ValueError(f"{event.ticker} already holds an event effective {effective}") from None
    return True, next(e for e in load_share_events(conn) if e.id == event_id)


def restatement(
    conn: sqlite3.Connection, ticker: str, published: Mapping[str, str | None]
) -> dict[str, Restated]:
    """Each per-share input restated to the share basis of the latest-published filing among
    them, given when each input's filing was published, by variable; only inputs that change.
    ValueError when an event between them is unchecked or unknown, or a publication date is not
    held while the company has events."""
    events = load_share_events(conn, ticker)
    if not events or len(published) < 2:
        return {}
    missing = sorted(v for v, day in published.items() if day is None)
    if missing:
        raise ValueError(
            f"{ticker} has share events, and when the filing of {', '.join(missing)} was "
            "published is not held; rebuild with python -m aigent.facts build --offline"
        )
    days = {v: str(day)[:10] for v, day in published.items()}
    latest = max(days.values())
    restated: dict[str, Restated] = {}
    for variable, day in sorted(days.items()):
        between = [e for e in events if _within(e, day, latest)]
        blocked = [e for e in between if not e.checked]
        if blocked:
            raise ValueError(
                f"{variable} was filed before a share-count change not yet checked: "
                f"{'; '.join(e.described() for e in blocked)}. Per-share figures across it "
                f"cannot be compared until a person checks it ({VERIFY_HOW})"
            )
        changing = [e for e in between if e.multiplier != 1.0]
        if changing:
            multiplier = math.prod(e.multiplier for e in changing)
            restated[variable] = Restated(multiplier, tuple(e.described() for e in changing))
    return restated
