"""`python -m aigent.facts build [--offline]`: build the facts database and write its report."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from aigent.facts.build import build
from aigent.facts.cfg_facts import FACTS_TICKERS, FACTS_YEARS, TAXONOMY_PACKAGES
from aigent.facts.nse import XBRL_DIR, polite_get
from aigent.facts.reference import load_fundamentals
from aigent.facts.store import FACTS_DB, Writer, connect
from aigent.facts.taxonomy import TAXONOMY_DIR

REPORT = FACTS_DB.with_name("facts-build-report.json")


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
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--offline", action="store_true", help="load the cache only; no fetching")
    args = parser.parse_args(argv)
    now = datetime.now().astimezone()
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


if __name__ == "__main__":
    main()
