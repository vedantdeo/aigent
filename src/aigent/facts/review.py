"""The cross-check's review queue: facts not agreed, grouped for one decision each, a person's
decisions read from `evals/reference/crosscheck-decisions.jsonl` and ruled against every run, so a
newly loaded filing is resolved by the same decisions; and the report."""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from aigent.facts.cfg_facts import (
    CROSSCHECK_GROUP_EXAMPLES,
    CROSSCHECK_SEED,
    CROSSCHECK_SPOT_CHECKS,
)
from aigent.facts.crosscheck import Checked, number

Issue = Literal["xbrl zero", "printed x100", "differs", "not printed"]
Verdict = Literal["right", "wrong", "elsewhere"]
Status = Literal["applied", "extend", "proposed", "refused"]
VERDICTS: tuple[Verdict, ...] = ("right", "wrong", "elsewhere")
ANY_COMPANY = "*"
DECISIONS = (
    Path(__file__).resolve().parents[3] / "evals" / "reference" / "crosscheck-decisions.jsonl"
)


@dataclass(frozen=True)
class Group:
    """Facts not agreed that share a company, concept, segment and issue, for one decision."""

    number: int
    ticker: str
    concept: str
    segment: str | None
    issue: Issue
    facts: tuple[Checked, ...]

    @property
    def id(self) -> str:
        """`TICKER|Concept|segment|issue`, the segment in lower case: what a decision names."""
        return f"{self.ticker}|{self.concept}|{(self.segment or '').casefold()}|{self.issue}"


@dataclass(frozen=True)
class Check:
    """A person's check under a `right` decision: the group it was made on, the facts it stands for
    with the value each had, and the spot facts they ticked."""

    group: str
    facts: tuple[tuple[int, float], ...]
    spot_checked: tuple[int, ...]


@dataclass(frozen=True)
class Decision:
    """A person's verdict on a group, or on every company's group when the id's company is `*`:
    `right` makes the facts its checks cover `xbrl`; `wrong`, and `elsewhere` (not printed in its
    results), take the group off the queue, now and as new filings add to it. With no `decided_by`
    it is only proposed."""

    id: str
    verdict: Verdict
    note: str
    decided_by: str | None = None
    checks: tuple[Check, ...] = ()


@dataclass(frozen=True)
class Ruling:
    """A decision against one of this run's groups. `extend`: a `right` decision whose checks cover
    only `confirmed`, with `new` facts awaiting a spot check of their own."""

    decision: Decision
    group: Group | None
    status: Status
    reason: str = ""
    confirmed: tuple[int, ...] = ()
    new: tuple[Checked, ...] = ()


def issue_of(c: Checked) -> Issue:
    """Why a fact did not agree: XBRL filed a zero, the printed figure is it times 100 (a ratio
    printed without its % sign), another printed figure differs, or none was found."""
    if c.value == 0:
        return "xbrl zero"
    if c.outcome != "differ":
        return "not printed"
    read = number(c.printed or "")
    if read is not None and abs(read[0] - c.value * 100) <= 0.5 * 10 ** -read[1] + 1e-9:
        return "printed x100"
    return "differs"


def review_groups(checked: Iterable[Checked]) -> list[Group]:
    """Every fact not agreed, grouped by company, concept, segment (any case) and issue, largest
    group first and numbered from 1."""
    by: dict[tuple[str, str, str, Issue], list[Checked]] = {}
    for c in checked:
        if c.outcome != "agree":
            key = (c.ticker, c.concept, (c.segment or "").casefold(), issue_of(c))
            by.setdefault(key, []).append(c)
    ranked = sorted(by.items(), key=lambda item: (-len(item[1]), item[0]))
    return [
        Group(i, ticker, concept, facts[0].segment, issue, tuple(facts))
        for i, ((ticker, concept, _, issue), facts) in enumerate(ranked, 1)
    ]


def review_sample(checked: Sequence[Checked], size: int, seed: int) -> list[Checked]:
    """A seeded random sample of the agreements, for a person to check."""
    agreed = [c for c in checked if c.outcome == "agree"]
    return sorted(
        random.Random(seed).sample(agreed, min(size, len(agreed))), key=lambda c: c.fact_id
    )


def spot_sample(
    group_id: str,
    fact_ids: Iterable[int],
    size: int = CROSSCHECK_SPOT_CHECKS,
    seed: int = CROSSCHECK_SEED,
) -> tuple[int, ...]:
    """The facts a person checks by hand before a check covers `fact_ids`; seeded by the group's
    id over the sorted ids, so a different set of facts draws a different sample."""
    ids = sorted(fact_ids)
    return tuple(sorted(random.Random(f"{seed}|{group_id}").sample(ids, min(size, len(ids)))))


def read_decisions(path: Path = DECISIONS) -> list[Decision]:
    """The decisions file, one JSON object a line, `//` lines comments; none when it is missing.
    Raises ValueError on an unknown field, a verdict not in `VERDICTS` or a malformed check, so a
    typo cannot pass for no decision."""
    if not path.exists():
        return []
    out: list[Decision] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("//"):
            continue
        where = f"{path.name}:{n}"
        given = _object(json.loads(line), {"id", "verdict", "note", "decided_by", "checks"}, where)
        verdict = given.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"{where}: verdict {verdict!r} is not one of {VERDICTS}")
        checks = given.get("checks", [])
        if not isinstance(checks, list):
            raise ValueError(f"{where}: checks must be a list")
        decided_by = given.get("decided_by")
        out.append(
            Decision(
                str(given["id"]),
                cast(Verdict, verdict),
                str(given.get("note", "")),
                None if decided_by is None else str(decided_by),
                tuple(_check(c, where) for c in cast(list[object], checks)),
            )
        )
    return out


def _object(value: object, fields: set[str], where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: not a JSON object")
    given = cast(dict[str, object], value)
    if unknown := set(given) - fields:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}")
    return given


def _check(value: object, where: str) -> Check:
    given = _object(value, {"group", "facts", "spot_checked"}, f"{where} check")
    facts, spot = given.get("facts"), given.get("spot_checked")
    pairs = cast(list[object], facts) if isinstance(facts, list) else None
    if pairs is None or not all(
        isinstance(p, list)
        and len(cast(list[object], p)) == 2
        and isinstance(cast(list[object], p)[0], int)
        and isinstance(cast(list[object], p)[1], int | float)
        for p in pairs
    ):
        raise ValueError(f"{where}: a check's facts must be [fact id, value] pairs")
    if not isinstance(spot, list) or not all(isinstance(i, int) for i in cast(list[object], spot)):
        raise ValueError(f"{where}: a check's spot_checked must be a list of fact ids")
    return Check(
        str(given.get("group", "")),
        tuple((int(p[0]), float(p[1])) for p in cast(list[list[float]], pairs)),
        tuple(cast(list[int], spot)),
    )


def rule(groups: Sequence[Group], decisions: Iterable[Decision]) -> list[Ruling]:
    """Each group against the decision naming it, its company's own before one for `*`. Proposed
    when nobody decided it; `wrong` and `elsewhere` applied to all its facts, old and new; `right`
    confirms the facts its checks cover at their checked values, refused when a check's ticks miss
    its spot sample or name a fact outside it, `extend` while other facts await a check. A decision
    naming one company's group that this run lacks is refused."""
    exact: dict[str, Decision] = {}
    every: dict[str, Decision] = {}
    for d in decisions:
        company, _, rest = d.id.partition("|")
        (every if company == ANY_COMPANY else exact)[rest if company == ANY_COMPANY else d.id] = d
    out: list[Ruling] = []
    for g in groups:
        d = exact.get(g.id) or every.get(g.id.partition("|")[2])
        if d is not None:
            out.append(_rule_one(g, d))
    present = {g.id for g in groups}
    out += [
        Ruling(d, None, "refused", "no group has this id in this run")
        for gid, d in exact.items()
        if gid not in present
    ]
    return out


def _rule_one(g: Group, d: Decision) -> Ruling:
    if d.decided_by is None:
        return Ruling(d, g, "proposed")
    if d.verdict != "right":
        return Ruling(d, g, "applied")
    now = {c.fact_id: c.value for c in g.facts}
    confirmed: set[int] = set()
    for check in d.checks:
        if check.group != g.id:
            continue
        ids = {i for i, _ in check.facts}
        missing = sorted(set(spot_sample(check.group, ids)) - set(check.spot_checked))
        outside = sorted(set(check.spot_checked) - ids)
        if missing or outside:
            why = "; ".join(
                part
                for part in (
                    f"spot check missing facts {missing}" if missing else "",
                    f"facts {outside} are not in the check" if outside else "",
                )
                if part
            )
            return Ruling(d, g, "refused", why)
        confirmed |= {i for i, value in check.facts if now.get(i) == value}
    new = tuple(c for c in g.facts if c.fact_id not in confirmed)
    if not new:
        return Ruling(d, g, "applied", confirmed=tuple(sorted(confirmed)))
    why = f"{len(new)} facts no check covers yet"
    return Ruling(d, g, "extend", why, tuple(sorted(confirmed)), new)


def apply_rulings(conn: sqlite3.Connection, rulings: Iterable[Ruling], now: datetime) -> int:
    """Makes each `xbrl-ai` fact a `right` decision's checks confirm `xbrl`, checked by the person
    who decided it; returns how many."""
    stamp = now.isoformat(timespec="seconds")
    moved = 0
    with conn:
        for r in rulings:
            if r.decision.verdict != "right" or r.status not in ("applied", "extend"):
                continue
            for fact_id in r.confirmed:
                cursor = conn.execute(
                    """UPDATE facts SET tier = 'xbrl', verified_by = ?, verified_at = ?
                       WHERE id = ? AND tier = 'xbrl-ai'""",
                    (r.decision.decided_by, stamp, fact_id),
                )
                moved += cursor.rowcount
    return moved


def write_report(
    checked: Sequence[Checked],
    sample: Sequence[Checked],
    header: Mapping[str, str],
    path: Path,
    rulings: Sequence[Ruling] = (),
) -> Path:
    """The counts per company, basis and outcome, the rulings, then the review queue: the groups
    still open, largest first (a group being extended shows only its new facts), each group's
    facts, and the sampled agreements; every row beside it as JSON lines."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for c in checked:
        counts.setdefault((c.ticker, c.basis), {"agree": 0, "differ": 0, "unmatched": 0})
        counts[(c.ticker, c.basis)][c.outcome] += 1
    by_group = {r.group.id: r for r in rulings if r.group is not None}
    queue: list[tuple[Group, Ruling | None, tuple[Checked, ...]]] = []
    for g in review_groups(checked):
        r = by_group.get(g.id)
        if r is None or r.status in ("proposed", "refused"):
            queue.append((g, r, g.facts))
        elif r.status == "extend":
            queue.append((g, r, r.new))
    lines = ["# XBRL cross-check", ""]
    lines += [f"- **{k}:** {v}" for k, v in header.items()]
    lines += ["", "| company | basis | agree | differ | unmatched |", "|---|---|---|---|---|"]
    for (ticker, basis), n in sorted(counts.items()):
        lines.append(f"| {ticker} | {basis} | {n['agree']} | {n['differ']} | {n['unmatched']} |")
    lines += ["", "## Decisions", ""]
    lines += [
        "| decision | group | verdict | status | by | facts | confirmed | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        _ruling_row(r)
        for r in sorted(rulings, key=lambda r: (r.decision.id, r.group.id if r.group else ""))
    ]
    lines += ["", "## Review queue: groups", ""]
    lines += [
        "| group | facts | company | concept | segment | issue | bases | periods | XBRL | printed "
        "| spot check | decision |"
    ]
    lines += ["|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [_group_row(g, r, facts) for g, r, facts in queue]
    lines += ["", "## Review queue: not agreed, by group", ""]
    lines += [
        "| group | fact | concept | segment | period | XBRL | printed | outcome | document p. |"
    ]
    lines += ["|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {g.number} {_row(c)}" for g, _, facts in queue for c in facts]
    lines += ["", "## Review queue: sampled agreements", ""]
    lines += ["| fact | concept | segment | period | XBRL | printed | outcome | document p. |"]
    lines += ["|---|---|---|---|---|---|---|---|"]
    lines += [_row(c) for c in sample]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = path.with_suffix(".rows.jsonl")
    rows.write_text("".join(json.dumps(asdict(c)) + "\n" for c in checked), encoding="utf-8")
    return path


def _ruling_row(r: Ruling) -> str:
    d = r.decision
    group = "" if r.group is None else _cell(r.group.id)
    facts = "" if r.group is None else str(len(r.group.facts))
    note = d.note if not r.reason else f"{r.reason}. {d.note}"
    return (
        f"| {_cell(d.id)} | {group} | {d.verdict} | {r.status} | {d.decided_by or ''} | {facts} | "
        f"{len(r.confirmed) if d.verdict == 'right' else ''} | {_cell(note)} |"
    )


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _group_row(g: Group, ruling: Ruling | None, facts: Sequence[Checked]) -> str:
    bases = ", ".join(sorted({c.basis for c in facts}))
    ends = sorted({c.period_end for c in facts})
    periods = ends[0] if len(ends) == 1 else f"{len(ends)}, {ends[0]} to {ends[-1]}"
    xbrl = _examples(f"{c.value:g} {c.unit}" for c in facts)
    printed = _examples(c.printed for c in facts if c.printed)
    spot = spot_sample(g.id, (c.fact_id for c in facts))
    return (
        f"| {g.number} | {len(facts)} | {g.ticker} | {g.concept} | {g.segment or ''} | "
        f"{g.issue} | {bases} | {periods} | {xbrl} | {printed} | "
        f"{', '.join(map(str, spot))} | {_cell(_pending(ruling))} |"
    )


def _pending(ruling: Ruling | None) -> str:
    if ruling is None:
        return ""
    d = ruling.decision
    if ruling.status == "refused":
        return f"{d.verdict} refused: {ruling.reason}"
    if ruling.status == "extend":
        return f"right by {d.decided_by}, {len(ruling.confirmed)} confirmed: check the new facts"
    return f"proposed {d.verdict}: {d.note}"


def _examples(values: Iterable[str]) -> str:
    """Distinct values in first-seen order, cut at `CROSSCHECK_GROUP_EXAMPLES`."""
    distinct = list(dict.fromkeys(values))
    shown = "; ".join(distinct[:CROSSCHECK_GROUP_EXAMPLES])
    return shown + "; …" if len(distinct) > CROSSCHECK_GROUP_EXAMPLES else shown


def _row(c: Checked) -> str:
    period = c.period_end if c.period_start is None else f"{c.period_start} to {c.period_end}"
    where = f"{c.document} p.{c.page}" if c.page else c.document
    return (
        f"| {c.fact_id} | {c.concept} | {c.segment or ''} | {period} | {c.value:g} {c.unit} | "
        f"{c.printed or ''} | {c.outcome} | {where} |"
    )
