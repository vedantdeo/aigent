"""The review queue: issues, groups, spot samples, decisions read and ruled against each run (new
facts included), what a ruling writes, and the report."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from aigent.facts import __main__ as cli
from aigent.facts.crosscheck import Checked
from aigent.facts.review import (
    Check,
    Decision,
    Group,
    apply_rulings,
    issue_of,
    read_decisions,
    review_groups,
    review_sample,
    rule,
    spot_sample,
    write_report,
)

from .conftest import checked_row

NOW = datetime(2026, 11, 1, 9, 0)


def _queued(
    fact_id: int,
    outcome: str = "unmatched",
    concept: str = "InterestEarned",
    segment: str | None = None,
    value: float = 1.0,
    ticker: str = "ICICIBANK",
) -> Checked:
    """A queued fact, its period ending on day `fact_id`; an unmatched one has no printed cell."""
    found = checked_row(fact_id, outcome)
    if outcome == "unmatched":
        found = replace(found, page=None, printed=None, row_label=None)
    return replace(
        found,
        concept=concept,
        segment=segment,
        value=value,
        ticker=ticker,
        period_end=f"2024-06-{fact_id:02d}",
    )


def _group(*ids: int, ticker: str = "ICICIBANK", value: float = 1.0) -> Group:
    facts = tuple(_queued(i, concept="CET1Ratio", value=value, ticker=ticker) for i in ids)
    return Group(1, ticker, "CET1Ratio", None, "not printed", facts)


def _check(
    group: Group, ids: range | tuple[int, ...], spot: tuple[int, ...] | None = None
) -> Check:
    """A check over `ids` at the group's values, its spot sample ticked unless `spot` is given."""
    values = {c.fact_id: c.value for c in group.facts}
    facts = tuple((i, values.get(i, 1.0)) for i in ids)
    ticked = spot_sample(group.id, ids) if spot is None else spot
    return Check(group.id, facts, ticked)


def test_the_review_sample_is_seeded_agreements_only() -> None:
    checked = [checked_row(i, "agree" if i % 2 else "differ") for i in range(1, 41)]
    first, again = review_sample(checked, 5, 7), review_sample(checked, 5, 7)
    assert first == again and len(first) == 5
    assert all(c.outcome == "agree" for c in first)
    assert len(review_sample(checked, 100, 7)) == 20


@pytest.mark.parametrize(
    ("outcome", "value", "issue"),
    [
        pytest.param("unmatched", 0.0, "xbrl zero", id="a zero nothing printed matches"),
        pytest.param("differ", 0.0, "xbrl zero", id="a zero beside a printed figure"),
        pytest.param("differ", 5.0, "differs", id="a figure the printed one differs from"),
        pytest.param("differ", 0.0149, "printed x100", id="a ratio printed without its % sign"),
        pytest.param("differ", 0.0049, "differs", id="a ratio printed as another"),
        pytest.param("unmatched", 5.0, "not printed", id="a figure nothing printed matches"),
    ],
)
def test_issue_of(outcome: str, value: float, issue: str) -> None:
    printed = {0.0149: "1.49", 0.0049: "0.97"}.get(value, "38,995.78")
    found = replace(
        _queued(1, outcome, value=value), printed=printed if outcome == "differ" else None
    )
    assert issue_of(found) == issue, found


def test_review_groups_leave_out_agreements_and_rank_largest_first() -> None:
    checked = [
        _queued(1, "agree"),
        _queued(2, concept="CET1Ratio"),
        _queued(3, concept="SegmentAssets", segment="Treasury", outcome="differ"),
        _queued(4, concept="SegmentAssets", segment="treasury", outcome="differ"),
        _queued(5, concept="CET1Ratio", value=0.0),
        _queued(6, concept="CET1Ratio"),
        _queued(7, concept="CET1Ratio", ticker="HDFCBANK"),
    ]
    shape = [
        (g.number, g.ticker, g.concept, g.segment, g.issue, [c.fact_id for c in g.facts])
        for g in review_groups(checked)
    ]
    assert shape == [
        (1, "ICICIBANK", "CET1Ratio", None, "not printed", [2, 6]),
        (2, "ICICIBANK", "SegmentAssets", "Treasury", "differs", [3, 4]),
        (3, "HDFCBANK", "CET1Ratio", None, "not printed", [7]),
        (4, "ICICIBANK", "CET1Ratio", None, "xbrl zero", [5]),
    ], shape


def test_a_groups_id_names_its_segment_in_lower_case() -> None:
    assert _group(1).id == "ICICIBANK|CET1Ratio||not printed"
    treasury = Group(1, "ICICIBANK", "SegmentAssets", "Treasury", "differs", ())
    assert treasury.id == "ICICIBANK|SegmentAssets|treasury|differs"


def test_the_spot_sample_is_seeded_and_drawn_from_the_facts_given() -> None:
    sample = spot_sample("G", range(1, 21), 3, 7)
    assert sample == spot_sample("G", reversed(range(1, 21)), 3, 7) and len(sample) == 3
    assert set(sample) <= set(range(1, 21)) and list(sample) == sorted(sample)
    assert len({spot_sample(f"G{i}", range(1, 21), 3, 7) for i in range(10)}) > 1
    assert spot_sample("G", (4, 9), 3, 7) == (4, 9)


def test_read_decisions_parses_rows_and_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    rows = [
        {
            "id": "A|B||differs",
            "verdict": "right",
            "note": "n",
            "decided_by": "vedant",
            "checks": [{"group": "A|B||differs", "facts": [[3, 1.5], [1, 2]], "spot_checked": [3]}],
        },
        {"id": "*|C||xbrl zero", "verdict": "wrong", "note": "filler"},
    ]
    path.write_text("// a comment\n\n" + "".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    assert read_decisions(path) == [
        Decision(
            "A|B||differs",
            "right",
            "n",
            "vedant",
            (Check("A|B||differs", ((3, 1.5), (1, 2.0)), (3,)),),
        ),
        Decision("*|C||xbrl zero", "wrong", "filler"),
    ]
    assert read_decisions(tmp_path / "missing.jsonl") == []


_CHECK = '{"group": "G", "facts": [[1, 2.0]], "spot_checked": [1]}'


@pytest.mark.parametrize(
    ("line", "message"),
    [
        pytest.param('{"id": "A", "verdict": "rigth"}', "verdict", id="a misspelt verdict"),
        pytest.param(
            '{"id": "A", "verdict": "right", "decided by": "v"}', "unknown", id="a misspelt field"
        ),
        pytest.param('["A", "right"]', "object", id="a row not an object"),
        pytest.param(
            '{"id": "A", "verdict": "right", "checks": {}}', "list", id="checks not a list"
        ),
        pytest.param(
            '{"id": "A", "verdict": "right", "checks": [' + _CHECK.replace("group", "grp") + "]}",
            "unknown",
            id="a misspelt check field",
        ),
        pytest.param(
            '{"id": "A", "verdict": "right", "checks": ['
            + _CHECK.replace("[[1, 2.0]]", "[1]")
            + "]}",
            "pairs",
            id="a check's facts not pairs",
        ),
        pytest.param(
            '{"id": "A", "verdict": "right", "checks": [' + _CHECK.replace("[1]}", '"1"}') + "]}",
            "spot_checked",
            id="a check's spot facts not a list",
        ),
    ],
)
def test_read_decisions_refuses_a_malformed_row(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_decisions(path)


_G = _group(*range(1, 11))
_SPOT = spot_sample(_G.id, range(1, 11))
_GROWN = _group(*range(1, 13))
_CHANGED = Group(
    1, _G.ticker, _G.concept, None, "not printed", (*_G.facts[:9], replace(_G.facts[9], value=7.0))
)


def _right(*checks: Check, by: str | None = "vedant") -> Decision:
    return Decision(_G.id, "right", "n", by, checks)


@pytest.mark.parametrize(
    ("group", "decision", "status", "confirmed", "new", "reason"),
    [
        pytest.param(
            _G,
            _right(_check(_G, range(1, 11)), by=None),
            "proposed",
            0,
            0,
            "",
            id="nobody decided it",
        ),
        pytest.param(
            _G, Decision(_G.id, "wrong", "n", "v"), "applied", 0, 0, "", id="wrong needs no check"
        ),
        pytest.param(
            _G,
            Decision(_G.id, "elsewhere", "n", "v"),
            "applied",
            0,
            0,
            "",
            id="elsewhere needs no check",
        ),
        pytest.param(_G, _right(), "extend", 0, 10, "10 facts", id="right with no check yet"),
        pytest.param(
            _G,
            _right(_check(_G, range(1, 11))),
            "applied",
            10,
            0,
            "",
            id="right with its spot sample ticked",
        ),
        pytest.param(
            _G,
            _right(_check(_G, range(1, 11), _SPOT[1:])),
            "refused",
            0,
            0,
            f"missing facts [{_SPOT[0]}]",
            id="a spot fact unticked",
        ),
        pytest.param(
            _G,
            _right(_check(_G, range(1, 11), (*_SPOT, 99))),
            "refused",
            0,
            0,
            "[99] are not in the check",
            id="a tick outside the check",
        ),
        pytest.param(
            _GROWN,
            _right(_check(_G, range(1, 11))),
            "extend",
            10,
            2,
            "2 facts",
            id="a new filing adds two facts",
        ),
        pytest.param(
            _GROWN,
            _right(_check(_G, range(1, 11)), _check(_GROWN, (11, 12))),
            "applied",
            12,
            0,
            "",
            id="the new facts checked too",
        ),
        pytest.param(
            _CHANGED,
            _right(_check(_G, range(1, 11))),
            "extend",
            9,
            1,
            "1 facts",
            id="a checked fact whose value changed",
        ),
        pytest.param(
            _G,
            _right(replace(_check(_G, range(1, 11)), group="HDFCBANK|CET1Ratio||not printed")),
            "extend",
            0,
            10,
            "10 facts",
            id="a check made on another group",
        ),
    ],
)
def test_rule(
    group: Group, decision: Decision, status: str, confirmed: int, new: int, reason: str
) -> None:
    [ruling] = rule([group], [decision])
    assert ruling.status == status, ruling
    assert (len(ruling.confirmed), len(ruling.new)) == (confirmed, new), ruling
    assert reason in ruling.reason and bool(ruling.reason) == bool(reason), ruling.reason


def test_a_decision_for_every_company_covers_each_and_yields_to_a_companys_own() -> None:
    icici, hdfc, other = _group(1, 2), _group(3, ticker="HDFCBANK"), _group(4, ticker="SBIN")
    decisions = [
        Decision("*|CET1Ratio||not printed", "wrong", "filler", "vedant"),
        Decision(hdfc.id, "right", "own", "vedant", (_check(hdfc, (3,)),)),
        Decision("*|Unseen||xbrl zero", "wrong", "no group yet", "vedant"),
    ]
    rulings = rule([icici, hdfc, other], decisions)
    shape = sorted((r.group.id if r.group else "", r.decision.id, r.status) for r in rulings)
    assert shape == [
        ("HDFCBANK|CET1Ratio||not printed", hdfc.id, "applied"),
        ("ICICIBANK|CET1Ratio||not printed", "*|CET1Ratio||not printed", "applied"),
        ("SBIN|CET1Ratio||not printed", "*|CET1Ratio||not printed", "applied"),
    ], shape


def test_a_companys_decision_naming_no_group_in_the_run_is_refused() -> None:
    [ruling] = rule([], [Decision("X|Y||differs", "wrong", "stale", "v")])
    assert (ruling.status, ruling.group) == ("refused", None)
    assert "no group" in ruling.reason


def test_apply_rulings_makes_only_confirmed_facts_xbrl(crosscheck_db: sqlite3.Connection) -> None:
    ids = [r[0] for r in crosscheck_db.execute("SELECT id FROM facts ORDER BY id")]
    crosscheck_db.execute(
        "UPDATE facts SET tier = 'xbrl-pdf', verified_by = 'pdf:d', verified_at = 't' WHERE id = ?",
        (ids[1],),
    )
    whole = Group(1, "ICICIBANK", "Whole", None, "not printed", tuple(_queued(i) for i in ids[:2]))
    grown = Group(2, "ICICIBANK", "Grown", None, "not printed", tuple(_queued(i) for i in ids[2:4]))
    wrong = Group(3, "ICICIBANK", "Wrong", None, "not printed", (_queued(ids[4]),))
    rulings = rule(
        [whole, grown, wrong],
        [
            Decision(whole.id, "right", "n", "vedant", (_check(whole, tuple(ids[:2])),)),
            Decision(grown.id, "right", "n", "vedant", (_check(grown, (ids[2],)),)),
            Decision(wrong.id, "wrong", "n", "vedant"),
        ],
    )
    assert apply_rulings(crosscheck_db, rulings, NOW) == 2
    rows = crosscheck_db.execute(
        "SELECT tier, verified_by, verified_at FROM facts ORDER BY id"
    ).fetchall()
    assert rows == [
        ("xbrl", "vedant", "2026-11-01T09:00:00"),
        ("xbrl-pdf", "pdf:d", "t"),
        ("xbrl", "vedant", "2026-11-01T09:00:00"),
        ("xbrl-ai", None, None),
        ("xbrl-ai", None, None),
    ], rows


def test_the_report_groups_the_queue_and_lists_each_groups_facts(tmp_path: Path) -> None:
    checked = [
        _queued(1, "agree"),
        _queued(2, value=0.0),
        *[_queued(i, concept="CET1Ratio", value=0.15 + i / 100) for i in range(3, 8)],
    ]
    path = write_report(checked, [checked[0]], {"run": "now"}, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    head, rest = text.split("## Review queue: groups")
    grouped, rest = rest.split("## Review queue: not agreed, by group")
    queued, sampled = rest.split("## Review queue: sampled agreements")
    assert "| ICICIBANK | standalone | 1 | 0 | 6 |" in head
    assert (
        "| 1 | 5 | ICICIBANK | CET1Ratio |  | not printed | standalone "
        "| 5, 2024-06-03 to 2024-06-07 | 0.18 INR; 0.19 INR; 0.2 INR; … |  |" in grouped
    ), grouped
    assert (
        "| 2 | 1 | ICICIBANK | InterestEarned |  | xbrl zero | standalone | 2024-06-02 |" in grouped
    )
    facts = [
        line.split(" | ")[:2]
        for line in queued.splitlines()
        if line.startswith("| ") and "---" not in line
    ]
    assert facts[1:] == [["| 1", str(i)] for i in range(3, 8)] + [["| 2", "2"]]
    assert "| 1 | InterestEarned |" in sampled
    rows = path.with_suffix(".rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(r)["fact_id"] for r in rows] == [1, 2, 3, 4, 5, 6, 7]


def test_the_report_keeps_open_and_extended_groups_on_the_queue(tmp_path: Path) -> None:
    checked = [
        *[_queued(i, concept="Decided") for i in (1, 2)],
        *[_queued(i, concept="Proposed") for i in (3, 4, 5)],
        *[_queued(i, concept="Grown") for i in (6, 7, 8, 9)],
        _queued(10, concept="Open"),
    ]
    groups = {g.concept: g for g in review_groups(checked)}
    grown = groups["Grown"]
    rulings = rule(
        list(groups.values()),
        [
            Decision(groups["Decided"].id, "wrong", "filler | zero", "vedant"),
            Decision(groups["Proposed"].id, "right", "printed in the label"),
            Decision(grown.id, "right", "n", "vedant", (_check(grown, (6, 7)),)),
            Decision("X|Y||differs", "wrong", "stale", "vedant"),
        ],
    )
    path = write_report(checked, [], {"run": "now"}, tmp_path / "r.md", rulings)
    text = path.read_text(encoding="utf-8")
    decisions, rest = text.split("## Decisions")[1].split("## Review queue: groups")
    grouped, queued = rest.split("## Review queue: not agreed, by group")
    queued = queued.split("## Review queue: sampled")[0]
    assert "| filler \\| zero |" in decisions and "| applied | vedant | 2 |" in decisions
    assert "| X\\|Y\\|\\|differs |  | wrong | refused |" in decisions, decisions
    assert "| right | extend | vedant | 4 | 2 |" in decisions, decisions
    assert "Decided" not in grouped and "Decided" not in queued
    spot = ", ".join(map(str, spot_sample(groups["Proposed"].id, (3, 4, 5))))
    assert f"| {spot} | proposed right: printed in the label |" in grouped, grouped
    new_spot = ", ".join(map(str, spot_sample(grown.id, (8, 9))))
    assert "| 2 | ICICIBANK | Grown |" in grouped
    assert f"| {new_spot} | right by vedant, 2 confirmed: check the new facts |" in grouped, grouped
    grown_rows = [line for line in queued.splitlines() if "| Grown |" in line]
    assert [r.split(" | ")[1] for r in grown_rows] == ["8", "9"], grown_rows
    assert "| ICICIBANK | Open |" in grouped


@pytest.mark.parametrize(("write", "tier"), [(False, "xbrl-ai"), (True, "xbrl")])
def test_the_command_applies_a_right_decision_only_when_told(
    crosscheck_db: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write: bool,
    tier: str,
) -> None:
    db = Path(crosscheck_db.execute("PRAGMA database_list").fetchone()[2])
    first = crosscheck_db.execute("SELECT min(id) FROM facts").fetchone()[0]
    crosscheck_db.close()
    [group] = review_groups([checked_row(first, "unmatched")])
    check = {"group": group.id, "facts": [[first, 1.0]], "spot_checked": [first]}
    row = {
        "id": group.id,
        "verdict": "right",
        "note": "n",
        "decided_by": "vedant",
        "checks": [check],
    }
    decisions = tmp_path / "d.jsonl"
    decisions.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "FACTS_DB", db)
    monkeypatch.setattr(cli, "CROSSCHECK_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli, "crosscheck", lambda c, t: [checked_row(first, "unmatched")])
    cli.cross_check(NOW, write=write, decisions=decisions)
    after = sqlite3.connect(db).execute("SELECT tier FROM facts WHERE id = ?", (first,)).fetchone()
    assert after == (tier,)
