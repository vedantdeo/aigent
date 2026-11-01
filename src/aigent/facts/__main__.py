"""`python -m aigent.facts build [--offline]`: build the facts database and write its report.
`python -m aigent.facts coverage`: score what it holds against the labelled figures; free.
`python -m aigent.facts crosscheck [--write]`: check the cross-checked companies' XBRL facts against
their results documents and write the review queue; `--write` makes agreements `xbrl-pdf`."""

from __future__ import annotations

import argparse
import json
import subprocess
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
from aigent.facts.crosscheck import apply, crosscheck, review_sample
from aigent.facts.crosscheck import write_report as write_crosscheck_report
from aigent.facts.lookup import Lookup, concept_index
from aigent.facts.nse import XBRL_DIR, polite_get
from aigent.facts.reference import load_fundamentals
from aigent.facts.store import FACTS_DB, Writer, connect
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
    parser.add_argument("command", choices=["build", "coverage", "crosscheck"])
    parser.add_argument("--offline", action="store_true", help="load the cache only; no fetching")
    parser.add_argument("--write", action="store_true", help="crosscheck: record agreements")
    args = parser.parse_args(argv)
    now = datetime.now().astimezone()
    if args.command == "coverage":
        coverage(now)
        return
    if args.command == "crosscheck":
        cross_check(now, write=args.write)
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


def cross_check(now: datetime, write: bool) -> None:
    """Check every cross-checked company's XBRL facts, write the report and review queue, and with
    `write`, make the agreements `xbrl-pdf`."""
    conn = connect(FACTS_DB)
    try:
        checked = crosscheck(conn, VERIFY_TICKERS)
        moved = apply(conn, checked, now) if write else 0
    finally:
        conn.close()
    agreed = sum(c.outcome == "agree" for c in checked)
    header = {
        "companies": ", ".join(VERIFY_TICKERS),
        "facts": f"{len(checked):,}, {agreed:,} agreeing with their results document",
        "written": f"{moved:,} made xbrl-pdf" if write else "no (run with --write to record)",
        "commit": _commit(),
        "run": f"{now:%Y-%m-%d %H:%M}",
        "cost": "$0.00 (local parsing only)",
    }
    sample = review_sample(checked, CROSSCHECK_SAMPLE, CROSSCHECK_SEED)
    path = write_crosscheck_report(
        checked, sample, header, CROSSCHECK_DIR / f"crosscheck-{now:%Y%m%d-%H%M}.md"
    )
    print(f"{len(checked)} facts, {agreed} agree, {moved} written; report: {path}")


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
