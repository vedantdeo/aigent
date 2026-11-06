"""`python -m aigent.facts build [--offline]`: build the facts database and write its report.
`python -m aigent.facts coverage`: score what it holds against the labelled figures; free.
`python -m aigent.facts crosscheck [--write]`: check the cross-checked companies' XBRL facts against
their results documents, rule on the review decisions and write the review queue; `--write` makes
agreements `xbrl-filed` and the groups decided `right` `xbrl`.
`python -m aigent.facts verify-concept --name N --scope S --by P`: a person's check of a derived
concept. `python -m aigent.facts share-events` lists the share events held, and
`verify-share-event --id N --by P [--kind K --effective D --ratio A:B]` records a person's check
of one, naming an unknown one's kind, date and ratio."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from aigent.facts.build import build
from aigent.facts.cfg_facts import (
    CROSSCHECK_SAMPLE,
    CROSSCHECK_SEED,
    FACTS_MATCH_METHOD,
    FACTS_TICKERS,
    FACTS_YEARS,
    TAXONOMY_PACKAGES,
    VERIFY_TICKERS,
)
from aigent.facts.coverage import (
    evaluate,
    fundamentals_figures,
    table_figures,
    teacher_figures,
    write_report,
)
from aigent.facts.crosscheck import apply, crosscheck
from aigent.facts.documents import origin, read_manifest
from aigent.facts.lookup import Lookup, concept_index
from aigent.facts.nse import XBRL_DIR, polite_get
from aigent.facts.reference import load_fundamentals
from aigent.facts.review import (
    DECISIONS,
    apply_rulings,
    read_decisions,
    review_groups,
    review_sample,
    rule,
)
from aigent.facts.review import write_report as write_crosscheck_report
from aigent.facts.share_events import load_share_events, verify_share_event
from aigent.facts.store import FACTS_DB, Writer, connect, verify_derived_concept
from aigent.facts.taxonomy import TAXONOMY_DIR

REPORT = FACTS_DB.with_name("facts-build-report.json")
CROSSCHECK_DIR = Path(__file__).resolve().parents[3] / "evals" / "reports"


def _commit() -> str:
    found = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
        check=False,
    )
    return found.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m aigent.facts")
    parser.add_argument(
        "command",
        choices=[
            "build",
            "coverage",
            "crosscheck",
            "verify-concept",
            "share-events",
            "verify-share-event",
        ],
    )
    parser.add_argument("--name", help="verify-concept: the derived concept checked")
    parser.add_argument("--scope", help="verify-concept: its scope, banking or ind-as")
    parser.add_argument("--by", help="verify-concept, verify-share-event: who checked it")
    parser.add_argument("--id", type=int, help="verify-share-event: the event, from share-events")
    parser.add_argument("--kind", help="verify-share-event: bonus, split or issue (for value)")
    parser.add_argument("--effective", help="verify-share-event: when the shares changed, ISO")
    parser.add_argument(
        "--ratio", help="verify-share-event: shares after:before, 2:1 for a 1:1 bonus"
    )
    parser.add_argument("--offline", action="store_true", help="load the cache only; no fetching")
    parser.add_argument(
        "--write", action="store_true", help="crosscheck: record agreements and decisions"
    )
    args = parser.parse_args(argv)
    now = datetime.now().astimezone()
    if args.command == "coverage":
        coverage(now)
        return
    if args.command == "crosscheck":
        cross_check(now, write=args.write)
        return
    if args.command == "verify-concept":
        verify_concept(str(args.name or ""), str(args.scope or ""), str(args.by or ""), now)
        return
    if args.command == "share-events":
        list_share_events()
        return
    if args.command == "verify-share-event":
        if args.id is None:
            raise SystemExit("give the event's --id, from share-events")
        verify_event(int(args.id), str(args.by or ""), now, args.kind, args.effective, args.ratio)
        return
    writer = Writer(
        written_by=f"xbrl-loader@{_commit()}",
        written_at=now.isoformat(timespec="seconds"),
        run_id=f"build-{now:%Y%m%dT%H%M%S}",
    )
    conn = connect(FACTS_DB)
    report = build(
        conn,
        XBRL_DIR,
        None if args.offline else polite_get(),
        FACTS_TICKERS,
        FACTS_YEARS,
        writer,
        now,
        load_fundamentals().values(),
        TAXONOMY_DIR,
        TAXONOMY_PACKAGES,
    )
    REPORT.write_text(json.dumps(report.as_json(), indent=1, default=str) + "\n", encoding="utf-8")
    print(report.summary())
    print(f"report: {REPORT}")


def verify_concept(name: str, scope: str, by: str, now: datetime) -> None:
    """Records a person's check of a derived concept's bindings, after showing them."""
    conn = connect(FACTS_DB)
    try:
        row = conn.execute(
            "SELECT label, formula, bindings, tier FROM derived_concepts "
            "WHERE name = ? AND scope = ?",
            (name, scope),
        ).fetchone()
        if row is None:
            raise SystemExit(f"no derived concept {name!r} in scope {scope!r}")
        print(f"{name} [{scope}]: {row[0]}\n  {row[1]} {row[2]}\n  was {row[3]}")
        try:
            changed, promoted = verify_derived_concept(
                conn, name, scope, by, now.isoformat(timespec="seconds")
            )
        except ValueError as bad:
            raise SystemExit(str(bad)) from None
    finally:
        conn.close()
    print(f"now derived-concept, checked by {by}" if changed else "already checked; unchanged")
    print(f"{promoted} computed figures made derived")


def list_share_events() -> None:
    """Prints every share event held, unchecked ones marked, with where each came from."""
    conn = connect(FACTS_DB)
    try:
        events = load_share_events(conn)
    finally:
        conn.close()
    for event in events:
        print(event.described() + (f"\n  {event.note}" if event.note else ""))
    print(f"{len(events)} events, {sum(not e.checked for e in events)} not checked")


def verify_event(
    event_id: int,
    by: str,
    now: datetime,
    kind: str | None = None,
    effective: str | None = None,
    ratio: str | None = None,
) -> None:
    """Records a person's check of a share event, after showing what it held."""
    conn = connect(FACTS_DB)
    try:
        held = [e for e in load_share_events(conn) if e.id == event_id]
        if held:
            print(f"was {held[0].described()}")
        try:
            changed, event = verify_share_event(
                conn, event_id, by, now.isoformat(timespec="seconds"), kind, effective, ratio
            )
        except ValueError as bad:
            raise SystemExit(str(bad)) from None
    finally:
        conn.close()
    print(f"now {event.described()}" if changed else "already checked as given; unchanged")


def cross_check(now: datetime, write: bool, decisions: Path = DECISIONS) -> None:
    """Check every cross-checked company's XBRL facts, rule on the decisions, write the report and
    review queue, and with `write`, make the agreements `xbrl-filed` and `right` groups `xbrl`."""
    conn = connect(FACTS_DB)
    try:
        checked = crosscheck(conn, VERIFY_TICKERS)
        rulings = rule(review_groups(checked), read_decisions(decisions))
        origins = {name: origin(entry) for name, entry in read_manifest().items()}
        moved = apply(conn, checked, now, origins) if write else 0
        checked_by_person = apply_rulings(conn, rulings, now) if write else 0
    finally:
        conn.close()
    agreed = sum(c.outcome == "agree" for c in checked)
    status = Counter(r.status for r in rulings)
    header = {
        "companies": ", ".join(VERIFY_TICKERS),
        "facts": f"{len(checked):,}, {agreed:,} agreeing with their results document",
        "decisions": (
            f"{status['applied']} applied, {status['extend']} awaiting a check of new facts, "
            f"{status['proposed']} proposed, {status['refused']} refused"
        ),
        "written": (
            f"{moved:,} made xbrl-filed, {checked_by_person:,} made xbrl"
            if write
            else "no (run with --write to record)"
        ),
        "commit": _commit(),
        "run": f"{now:%Y-%m-%d %H:%M}",
        "cost": "$0.00 (local parsing only)",
    }
    sample = review_sample(checked, CROSSCHECK_SAMPLE, CROSSCHECK_SEED)
    path = write_crosscheck_report(
        checked, sample, header, CROSSCHECK_DIR / f"crosscheck-{now:%Y%m%d-%H%M}.md", rulings
    )
    print(
        f"{len(checked)} facts, {agreed} agree, {moved} written xbrl-filed, "
        f"{checked_by_person} written xbrl; decisions: {header['decisions']}; report: {path}"
    )


def coverage(now: datetime) -> None:
    """Score every source's figures and write the report; its lookups are not kept."""
    tables, left_tables = table_figures()
    teacher, left_teacher = teacher_figures()
    figures = [*fundamentals_figures(load_fundamentals().values()), *tables, *teacher]
    lookup = Lookup(
        FACTS_DB,
        concept_index(),
        run_id=f"coverage-{now:%Y%m%dT%H%M%S}",
        asked_by=f"coverage@{_commit()}",
    )
    conn = connect(FACTS_DB)
    try:
        filings, facts = conn.execute(
            "SELECT (SELECT COUNT(*) FROM filings), (SELECT COUNT(*) FROM facts)"
        ).fetchone()
        scored = evaluate(conn, lookup, figures)
    finally:
        conn.close()
    header = {
        "database": f"`{FACTS_DB.name}`, {filings} filings, {facts:,} facts",
        "matched by": f"`{FACTS_MATCH_METHOD}` over the concepts' labels",
        "commit": _commit(),
        "run": f"{now:%Y-%m-%d %H:%M}",
        "cost": "$0.00 (local models only)",
    }
    path = write_report(scored, [*left_tables, *left_teacher], header, now)
    print(f"{len(figures)} figures, {len(scored)} lookups; report: {path}")


if __name__ == "__main__":
    main()
