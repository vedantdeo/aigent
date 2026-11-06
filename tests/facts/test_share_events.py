"""`facts.share_events`: the schema's refusals, seeding, detecting an unknown change from the filed
share count, a person's check, and restating per-share figures across checked events only."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aigent.facts.share_events import (
    SHARE_EVENTS,
    detect_share_events,
    load_share_events,
    restatement,
    seed_share_events,
    share_counts,
    sync_share_events,
    verify_share_event,
)
from aigent.facts.store import Writer, connect

from ..conftest import share_event, share_filing

WRITER = Writer("xbrl-loader@abc", "2026-11-06T10:00:00+05:30", "build-1")
Event = tuple[str, str | None, tuple[int, int] | None, bool]  # kind, effective, ratio, checked


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return connect(tmp_path / "facts.sqlite")


def _counts(conn: sqlite3.Connection, *counts: tuple[str, float]) -> None:
    with conn:
        for end, shares in counts:
            share_filing(conn, "HDFCBANK", end, shares)


def _events(conn: sqlite3.Connection, *events: Event) -> None:
    with conn:
        for kind, effective, ratio, checked in events:
            share_event(conn, "HDFCBANK", kind, effective, ratio, checked)


@pytest.mark.parametrize(
    ("counts", "held", "unknown"),
    [
        pytest.param([("2025-06-30", 100.0), ("2025-09-30", 100.5)], [], [], id="a steady count"),
        pytest.param(
            [("2025-06-30", 100.0), ("2025-09-30", 104.0)], [], [], id="option drift under 5%"
        ),
        pytest.param(
            [("2025-06-30", 100.0), ("2025-09-30", 200.0)],
            [],
            [("2025-06-30", "2025-09-30", 2.0)],
            id="a doubling with no event held",
        ),
        pytest.param(
            [("2025-06-30", 100.0), ("2025-09-30", 200.0)],
            [("bonus", "2025-08-28", (2, 1), False)],
            [],
            id="a doubling with a bonus held inside, even unchecked",
        ),
        pytest.param(
            [("2025-06-30", 100.0), ("2025-09-30", 200.0)],
            [("bonus", "2025-09-30", (2, 1), True)],
            [],
            id="an event on the later period end is inside",
        ),
        pytest.param(
            [("2025-06-30", 100.0), ("2025-09-30", 200.0)],
            [("bonus", "2025-06-30", (2, 1), True)],
            [("2025-06-30", "2025-09-30", 2.0)],
            id="an event on the earlier period end is not",
        ),
        pytest.param(
            [("2025-06-30", 100.0), ("2025-09-30", 200.0)],
            [("issue", "2025-08-01", None, True)],
            [],
            id="an issue for value held inside",
        ),
        pytest.param(
            [("2025-06-30", 100.0), ("2025-09-30", 90.0)],
            [],
            [("2025-06-30", "2025-09-30", 0.9)],
            id="a fall, as from a buyback",
        ),
        pytest.param(
            [("2025-03-31", 100.0), ("2025-06-30", 100.0), ("2025-09-30", 133.0)],
            [],
            [("2025-06-30", "2025-09-30", 1.33)],
            id="only the pair that moved",
        ),
    ],
)
def test_a_share_count_moving_with_no_event_held_is_held_as_unknown(
    conn: sqlite3.Connection,
    counts: list[tuple[str, float]],
    held: list[Event],
    unknown: list[tuple[str, str, float]],
) -> None:
    _counts(conn, *counts)
    _events(conn, *held)
    made = detect_share_events(conn, WRITER)
    seen = [(e.seen_from, e.seen_to, round(e.seen_ratio or 0, 3)) for e in made]
    assert seen == unknown, [e.described() for e in made]
    assert all(e.kind == "unknown" and e.tier == "share-event-ai" for e in made)


def test_an_unknown_change_is_held_once(conn: sqlite3.Connection) -> None:
    _counts(conn, ("2025-06-30", 100.0), ("2025-09-30", 200.0))
    assert len(detect_share_events(conn, WRITER)) == 1
    assert detect_share_events(conn, WRITER) == []
    assert len(load_share_events(conn)) == 1


def test_the_share_count_is_the_standalone_filing_s_own(conn: sqlite3.Connection) -> None:
    with conn:
        share_filing(conn, "HDFCBANK", "2025-06-30", 100.0)
        share_filing(conn, "HDFCBANK", "2025-09-30", 300.0, basis="consolidated")
        share_filing(conn, "HDFCBANK", "2025-09-30", 101.0)
    assert share_counts(conn, "HDFCBANK") == [("2025-06-30", 100.0), ("2025-09-30", 101.0)]
    assert detect_share_events(conn, WRITER) == []


def test_the_shipped_events_seed_once_unchecked(conn: sqlite3.Connection) -> None:
    assert seed_share_events(conn, WRITER) == len(SHARE_EVENTS.read_text().splitlines()) == 4
    assert seed_share_events(conn, WRITER) == 0
    events = load_share_events(conn)
    assert {(e.ticker, e.effective, e.multiplier) for e in events} == {
        ("HDFCBANK", "2025-08-28", 0.5),
        ("POWERGRID", "2023-09-12", 0.75),
        ("RELIANCE", "2024-10-29", 0.5),
        ("WIPRO", "2024-12-04", 0.5),
    }
    assert {e.tier for e in events} == {"share-event-ai"} and all(e.note for e in events)


def test_a_seed_row_the_schema_refuses_raises(conn: sqlite3.Connection, tmp_path: Path) -> None:
    path = tmp_path / "share-events.jsonl"
    path.write_text(json.dumps({"ticker": "ITC", "effective": "2025-01-01", "kind": "bonus"}))
    with pytest.raises(sqlite3.IntegrityError):
        seed_share_events(conn, WRITER, path)


def test_a_sync_seeds_then_detects_around_the_seeded(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    path = tmp_path / "share-events.jsonl"
    path.write_text(
        json.dumps(
            {
                "ticker": "HDFCBANK",
                "effective": "2025-08-28",
                "kind": "bonus",
                "shares_after": 2,
                "shares_before": 1,
            }
        )
    )
    _counts(conn, ("2025-06-30", 100.0), ("2025-09-30", 200.0), ("2025-12-31", 300.0))
    seeded, unknown = sync_share_events(conn, WRITER, path)
    assert seeded == 1 and [(e.seen_from, e.seen_to) for e in unknown] == [
        ("2025-09-30", "2025-12-31")
    ]
    assert load_share_events(conn)[0].ticker == "HDFCBANK"


_COLUMNS = "ticker, kind, effective, shares_after, shares_before, seen_to, tier, verified_by"
_OK = ("HDFCBANK", "bonus", "2025-01-15", 2, 1, None, "share-event-ai", None)  # another day


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            ("HDFCBANK", "unknown", "2025-08-28", None, None, "2025-09-30", "share-event-ai", None),
            id="an unknown change with a date",
        ),
        pytest.param(
            ("HDFCBANK", "unknown", None, None, None, "2025-09-30", "share-event", "vedant"),
            id="an unknown change checked",
        ),
        pytest.param(
            ("HDFCBANK", "unknown", None, None, None, None, "share-event-ai", None),
            id="an unknown change with no window",
        ),
        pytest.param(
            ("HDFCBANK", "bonus", "2025-08-28", None, None, None, "share-event-ai", None),
            id="a bonus with no ratio",
        ),
        pytest.param(
            ("HDFCBANK", "issue", "2025-08-28", 2, 1, None, "share-event-ai", None),
            id="an issue with a ratio",
        ),
        pytest.param(
            ("HDFCBANK", "bonus", "2025-08-28", 2, None, None, "share-event-ai", None),
            id="half a ratio",
        ),
        pytest.param(
            ("HDFCBANK", "split", "2025-08-28", 5, 5, None, "share-event-ai", None),
            id="a ratio changing nothing",
        ),
        pytest.param(
            ("HDFCBANK", "split", "2025-08-28", 0, 1, None, "share-event-ai", None),
            id="no shares after",
        ),
        pytest.param(
            ("HDFCBANK", "bonus", "2025-08-28", 2, 1, None, "share-event", None),
            id="checked with no checker",
        ),
        pytest.param(
            ("HDFCBANK", "bonus", "2025-08-28", 2, 1, None, "share-event-ai", "v"),
            id="a checker on an unchecked event",
        ),
        pytest.param(
            ("HDFCBANK", "rights", "2025-08-28", 2, 1, None, "share-event-ai", None),
            id="a kind not listed",
        ),
        pytest.param(_OK, id="a second event on the same date"),
    ],
)
def test_the_schema_refuses_what_cannot_be_an_event(
    conn: sqlite3.Connection, row: tuple[object, ...]
) -> None:
    insert = f"""INSERT INTO share_events ({_COLUMNS}, verified_at, written_by, written_at,
                 run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'w', 't', 'r')"""
    conn.execute(insert, (*_OK, None))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, (*row, "t" if row[-1] else None))


@pytest.mark.parametrize(
    ("start", "given", "held"),
    [
        pytest.param(
            ("bonus", "2025-08-28", (2, 1)),
            {},
            ("bonus", "2025-08-28", 2, 1),
            id="a seeded bonus confirmed",
        ),
        pytest.param(
            ("bonus", "2025-08-28", (2, 1)),
            {"ratio": "4:3", "effective": "2025-08-27"},
            ("bonus", "2025-08-27", 4, 3),
            id="a seeded bonus corrected",
        ),
        pytest.param(
            ("unknown", None, None),
            {"kind": "bonus", "effective": "2025-08-28", "ratio": "2:1"},
            ("bonus", "2025-08-28", 2, 1),
            id="an unknown change named a bonus",
        ),
        pytest.param(
            ("unknown", None, None),
            {"kind": "issue", "effective": "2025-09-30"},
            ("issue", "2025-09-30", None, None),
            id="an unknown change named an issue",
        ),
        pytest.param(
            ("bonus", "2025-08-28", (2, 1)),
            {"kind": "issue"},
            ("issue", "2025-08-28", None, None),
            id="a seeded bonus that was an issue",
        ),
    ],
)
def test_a_person_s_check_makes_an_event_checked(
    conn: sqlite3.Connection,
    start: tuple[str, str | None, tuple[int, int] | None],
    given: dict[str, str],
    held: tuple[str, str, int | None, int | None],
) -> None:
    kind, effective, ratio = start
    seen = ("2025-06-30", "2025-09-30") if kind == "unknown" else None
    with conn:
        event_id = share_event(conn, "HDFCBANK", kind, effective, ratio, seen=seen)
    changed, event = verify_share_event(conn, event_id, "vedant", "t2", **given)
    assert changed and event.tier == "share-event" and event.verified_by == "vedant", event
    assert (event.kind, event.effective, event.shares_after, event.shares_before) == held
    assert verify_share_event(conn, event_id, "vedant", "t3", **given) == (False, event)


@pytest.mark.parametrize(
    ("start", "given", "said"),
    [
        pytest.param(None, {}, "no share event", id="an event not held"),
        pytest.param(("bonus", "2025-08-28"), {"person": " "}, "who checked", id="no person"),
        pytest.param(
            ("unknown", None),
            {"effective": "2025-08-28"},
            "needs its kind",
            id="an unknown change with no kind",
        ),
        pytest.param(
            ("unknown", None),
            {"kind": "bonus", "ratio": "2:1"},
            "the date",
            id="an unknown change with no date",
        ),
        pytest.param(
            ("bonus", "2025-08-28"),
            {"kind": "unknown"},
            "bonus, split or issue",
            id="named unknown again",
        ),
        pytest.param(
            ("bonus", "2025-08-28"),
            {"kind": "rights"},
            "bonus, split or issue",
            id="a kind not listed",
        ),
        pytest.param(
            ("bonus", "2025-08-28"), {"effective": "28-08-2025"}, "not a date", id="a date not ISO"
        ),
        pytest.param(
            ("unknown", None),
            {"kind": "issue", "effective": "2025-10-01"},
            "outside",
            id="a date after the change was seen",
        ),
        pytest.param(
            ("unknown", None),
            {"kind": "issue", "effective": "2025-06-30"},
            "outside",
            id="a date on the period end before it",
        ),
        pytest.param(
            ("unknown", None),
            {"kind": "bonus", "effective": "2025-08-28"},
            "needs its ratio",
            id="a bonus with no ratio",
        ),
        pytest.param(
            ("bonus", "2025-08-28"),
            {"kind": "issue", "ratio": "2:1"},
            "give no ratio",
            id="an issue given a ratio",
        ),
        pytest.param(
            ("bonus", "2025-08-28"), {"ratio": "2-1"}, "after:before", id="a ratio not after:before"
        ),
        pytest.param(
            ("bonus", "2025-08-28"),
            {"ratio": "1:1"},
            "changes no share count",
            id="a ratio changing nothing",
        ),
        pytest.param(
            ("bonus", "2025-08-28"),
            {"effective": "2025-01-15"},
            "already holds",
            id="a date another event holds",
        ),
    ],
)
def test_a_check_that_cannot_describe_an_event_is_refused_and_changes_nothing(
    conn: sqlite3.Connection,
    start: tuple[str, str | None] | None,
    given: dict[str, str],
    said: str,
) -> None:
    with conn:
        share_event(conn, "HDFCBANK", "split", "2025-01-15", (5, 1))
        event_id = 99
        if start is not None:
            kind, effective = start
            seen = ("2025-06-30", "2025-09-30") if kind == "unknown" else None
            ratio = (2, 1) if kind == "bonus" else None
            event_id = share_event(conn, "HDFCBANK", kind, effective, ratio, seen=seen)
    before = load_share_events(conn)
    person = given.pop("person", "vedant")
    with pytest.raises(ValueError, match=said):
        verify_share_event(conn, event_id, person, "t2", **given)
    assert load_share_events(conn) == before


AFTER, BEFORE = "2025-10-18T08:22:38", "2024-10-19T12:01:00"


@pytest.mark.parametrize(
    ("events", "published", "restated"),
    [
        pytest.param([], {"current": AFTER, "previous": BEFORE}, {}, id="no events held"),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True)], {"current": AFTER}, {}, id="one input"
        ),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True)],
            {"current": AFTER, "previous": BEFORE},
            {"previous": 0.5},
            id="a checked 1:1 bonus between halves the earlier",
        ),
        pytest.param(
            [("bonus", "2025-08-28", (4, 3), True)],
            {"current": AFTER, "previous": BEFORE},
            {"previous": 0.75},
            id="a checked 1:3 bonus",
        ),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True), ("split", "2025-03-01", (5, 1), True)],
            {"current": AFTER, "previous": BEFORE},
            {"previous": 0.1},
            id="two checked events multiply",
        ),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True)],
            {"current": AFTER, "previous": "2025-10-01T09:00:00"},
            {},
            id="both published after",
        ),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True)],
            {"current": "2025-07-19T10:00:00", "previous": BEFORE},
            {},
            id="both published before",
        ),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True)],
            {"current": "2025-08-28T18:00:00", "previous": BEFORE},
            {"previous": 0.5},
            id="published the day it took effect counts as after",
        ),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True)],
            {"previous": AFTER, "current": BEFORE},
            {"current": 0.5},
            id="the latest filing's basis, whichever variable it is",
        ),
        pytest.param(
            [("issue", "2025-08-28", None, True)],
            {"current": AFTER, "previous": BEFORE},
            {},
            id="a checked issue restates nothing",
        ),
        pytest.param(
            [("bonus", "2023-09-12", (4, 3), False)],
            {"current": AFTER, "previous": BEFORE},
            {},
            id="an unchecked event before both is no concern",
        ),
    ],
)
def test_per_share_inputs_are_restated_across_checked_events(
    conn: sqlite3.Connection,
    events: list[Event],
    published: dict[str, str | None],
    restated: dict[str, float],
) -> None:
    _events(conn, *events)
    got = restatement(conn, "HDFCBANK", published)
    assert {v: pytest.approx(r.multiplier) for v, r in got.items()} == restated, got
    assert all(r.events for r in got.values())


@pytest.mark.parametrize(
    ("events", "published", "said"),
    [
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), False)],
            {"current": AFTER, "previous": BEFORE},
            "not yet checked",
            id="an unchecked bonus between",
        ),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True), ("split", "2025-03-01", (5, 1), False)],
            {"current": AFTER, "previous": BEFORE},
            "not yet checked",
            id="one of two events unchecked",
        ),
        pytest.param(
            [("bonus", "2025-08-28", (2, 1), True)],
            {"current": AFTER, "previous": None},
            "rebuild",
            id="a publication date not held",
        ),
    ],
)
def test_a_restatement_across_an_unchecked_event_is_refused(
    conn: sqlite3.Connection, events: list[Event], published: dict[str, str | None], said: str
) -> None:
    _events(conn, *events)
    with pytest.raises(ValueError, match=said):
        restatement(conn, "HDFCBANK", published)


@pytest.mark.parametrize(
    ("seen", "refused"),
    [
        pytest.param(("2025-06-30", "2025-09-30"), True, id="an unknown change between them"),
        pytest.param(("2024-09-30", "2024-12-31"), True, id="one the earlier filing may predate"),
        pytest.param(("2024-06-30", "2024-09-30"), False, id="one seen before the earlier was out"),
        pytest.param(("2023-06-30", "2023-09-30"), False, id="one before both"),
        pytest.param(("2025-12-31", "2026-03-31"), False, id="one after both"),
        pytest.param(("2025-10-18", "2025-12-31"), False, id="one after the latest was out"),
    ],
)
def test_an_unknown_change_refuses_any_restatement_it_may_fall_inside(
    conn: sqlite3.Connection, seen: tuple[str, str], refused: bool
) -> None:
    with conn:
        share_event(conn, "HDFCBANK", "unknown", None, seen=seen)
    published: dict[str, str | None] = {"current": AFTER, "previous": BEFORE}
    if refused:
        with pytest.raises(ValueError, match="unknown share-count change"):
            restatement(conn, "HDFCBANK", published)
    else:
        assert restatement(conn, "HDFCBANK", published) == {}


def test_the_commands_list_events_and_record_a_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from aigent.facts import __main__ as cli

    db = tmp_path / "facts.sqlite"
    conn = connect(db)
    with conn:
        bonus = share_event(conn, "HDFCBANK", "bonus", "2025-08-28", (2, 1))
        unknown = share_event(conn, "WIPRO", "unknown", None, seen=("2024-09-30", "2024-12-31"))
    conn.close()
    monkeypatch.setattr(cli, "FACTS_DB", db)
    cli.main(["share-events"])
    out = capsys.readouterr().out
    assert "HDFCBANK event" in out and "unknown share-count change" in out, out
    assert "2 events, 2 not checked" in out, out
    cli.main(["verify-share-event", "--id", str(bonus), "--by", "vedant"])
    out = capsys.readouterr().out
    assert "was HDFCBANK" in out and "now HDFCBANK" in out and "checked by vedant" in out, out
    cli.main(["verify-share-event", "--id", str(bonus), "--by", "vedant"])
    assert "unchanged" in capsys.readouterr().out
    named = ["--kind", "bonus", "--effective", "2024-12-04", "--ratio", "2:1"]
    cli.main(["verify-share-event", "--id", str(unknown), "--by", "vedant", *named])
    assert "bonus 2:1 effective 2024-12-04" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="needs its kind"):
        conn = connect(db)
        with conn:
            again = share_event(conn, "ITC", "unknown", None, seen=("2024-09-30", "2024-12-31"))
        conn.close()
        cli.main(["verify-share-event", "--id", str(again), "--by", "vedant"])
    with pytest.raises(SystemExit, match="--id"):
        cli.main(["verify-share-event", "--by", "vedant"])
