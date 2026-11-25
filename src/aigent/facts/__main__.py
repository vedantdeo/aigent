"""`python -m aigent.facts build [--offline]`: build the facts database and write its report.
`python -m aigent.facts coverage`: score what it holds against the labelled figures; free.
`python -m aigent.facts crosscheck [--recheck] [--write]`: check the cross-checked companies' XBRL
facts against their results documents, rule on the review decisions and write the review queue;
`--recheck` stamps afresh or demotes each `xbrl-filed` fact checked by an older reading, and
`--write` makes agreements `xbrl-filed` and the groups decided `right` `xbrl`.
`python -m aigent.facts reocr [--documents PATH ...] [--tickers T ...]`: OCR scanned results afresh,
each in its own process, into the readings the cross-check tries after them; free, minutes each.
`python -m aigent.facts verify-concept --name N --scope S --by P`: a person's check of a derived
concept. `python -m aigent.facts share-events` lists the share events held, and
`verify-share-event --id N --by P [--kind K --effective D --ratio A:B]` records a person's check
of one, naming an unknown one's kind, date and ratio.
`python -m aigent.facts trace --rows PATH --facts N ...`: each fact's cell from a cross-check run's
`.rows.jsonl`, traced through its table as parsed and as the cross-check read it, as JSON
on stdout; free, from the parse caches.
`python -m aigent.facts review-page --name N [--outcome agree|differ] [--add K] [--rows PATH]
[--url U]`: build review page N's HTML from a run's rows (the newest by default), first drawing K
more cards when asked, into `evals/review-pages/N/index.html` for Claude to publish; free."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast

import psutil

from aigent.facts.build import build
from aigent.facts.cfg_facts import (
    CROSSCHECK_SAMPLE,
    CROSSCHECK_SEED,
    FACTS_MATCH_METHOD,
    FACTS_TICKERS,
    FACTS_YEARS,
    REVIEW_PAGE_SEED,
    TAXONOMY_PACKAGES,
    VERIFY_REOCR_MAX_GB,
    VERIFY_REOCR_POLL_S,
    VERIFY_TICKERS,
)
from aigent.facts.coverage import (
    evaluate,
    fundamentals_figures,
    table_figures,
    teacher_figures,
    write_report,
)
from aigent.facts.crosscheck import Checked, Sheet, apply, crosscheck, recheck, sheets
from aigent.facts.documents import (
    Entry,
    build_reocr,
    filed_of,
    is_reading,
    load_verify_document,
    origin_of,
    read_manifest,
    readings_of,
    reocr_built,
    ticker_of,
)
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
from aigent.facts.review_page import (
    REVIEW_PAGES,
    TEMPLATE,
    PageList,
    PageOutcome,
    build_page,
    draw,
    kinds_of,
    page_cards,
    page_image,
    read_page_list,
    write_page_list,
)
from aigent.facts.share_events import load_share_events, verify_share_event
from aigent.facts.store import FACTS_DB, Writer, connect, verify_derived_concept
from aigent.facts.taxonomy import TAXONOMY_DIR
from aigent.facts.trace import trace, trace_json
from aigent.retrieval.chunk import Document

REPORT = FACTS_DB.with_name("facts-build-report.json")
CROSSCHECK_DIR = Path(__file__).resolve().parents[3] / "evals" / "reports"
REVIEW_PAGES_OUT = CROSSCHECK_DIR.parent / "review-pages"


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
            "reocr",
            "verify-concept",
            "share-events",
            "verify-share-event",
            "trace",
            "review-page",
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
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="crosscheck: stamp afresh or demote facts an older reading made xbrl-filed",
    )
    parser.add_argument(
        "--documents", nargs="*", default=[], help="reocr: scans by manifest path, as results/X.pdf"
    )
    parser.add_argument("--tickers", nargs="*", default=[], help="reocr: every scan of these")
    parser.add_argument("--rows", type=Path, help="trace: a cross-check run's .rows.jsonl")
    parser.add_argument("--facts", nargs="*", type=int, default=[], help="trace: fact ids")
    parser.add_argument("--outcome", choices=["agree", "differ"], help="review-page: a new page's")
    parser.add_argument("--add", type=int, default=0, help="review-page: cards to draw anew")
    parser.add_argument("--url", help="review-page: the artifact the page is published at")
    parser.add_argument("--one", help=argparse.SUPPRESS)  # reocr's child: build one reading
    args = parser.parse_args(argv)
    now = datetime.now().astimezone()
    if args.command == "coverage":
        coverage(now)
        return
    if args.command == "crosscheck":
        cross_check(now, write=args.write, stale=args.recheck)
        return
    if args.command == "reocr":
        if args.one:
            build_reocr(args.one)
            return
        reocr(args.documents, args.tickers)
        return
    if args.command == "verify-concept":
        verify_concept(str(args.name or ""), str(args.scope or ""), str(args.by or ""), now)
        return
    if args.command == "share-events":
        list_share_events()
        return
    if args.command == "trace":
        if args.rows is None or not args.facts:
            raise SystemExit("give a run's --rows file and the --facts to trace")
        print(json.dumps(trace_facts(args.rows, list(args.facts)), indent=1))
        return
    if args.command == "review-page":
        if not args.name:
            raise SystemExit("give the page's --name, as spot-check")
        outcome = cast(PageOutcome | None, args.outcome)
        review_page(str(args.name), outcome, args.rows, int(args.add), args.url)
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


def cross_check(
    now: datetime, write: bool, decisions: Path = DECISIONS, stale: bool = False
) -> None:
    """Check every cross-checked company's XBRL facts, rule on the decisions, write the report and
    review queue; with `stale`, re-check the `xbrl-filed` facts an older reading stamped, and with
    `write`, make the agreements `xbrl-filed` and `right` groups `xbrl`."""
    conn = connect(FACTS_DB)
    try:
        checked = crosscheck(conn, VERIFY_TICKERS)
        rulings = rule(review_groups(checked), read_decisions(decisions))
        manifest = read_manifest()
        origins = {r: origin_of(r, e) for n, e in manifest.items() for r in readings_of(n, e)}
        rechecked = recheck(conn, checked, now, origins, VERIFY_TICKERS) if stale else None
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
        "rechecked": (
            f"{rechecked.restamped:,} stamped afresh, {rechecked.demoted:,} made xbrl-ai, "
            f"{rechecked.derived_demoted:,} derived made derived-ai"
            if rechecked
            else "no (run with --recheck to re-check stale stamps)"
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
        f"{checked_by_person} written xbrl; rechecked: {header['rechecked']}; "
        f"decisions: {header['decisions']}; report: {path}"
    )
    missing = [r for r in origins if is_reading(r) and not reocr_built(r)]
    if missing:
        print(f"{len(missing)} scans have no re-OCR'd reading yet: python -m aigent.facts reocr")


def trace_facts(
    rows: Path,
    fact_ids: Sequence[int],
    load: Callable[[str], Document] = load_verify_document,
) -> list[dict[str, object]]:
    """Each fact's trace from a run's rows, in the order asked, each document read once; a fact not
    in the run, or one that cannot be traced, comes back as its id and the error."""
    wanted = set(fact_ids)
    found: dict[int, Checked] = {}
    for line in rows.read_text(encoding="utf-8").splitlines():
        fields = json.loads(line)
        if fields["fact_id"] in wanted:
            found[fields["fact_id"]] = Checked(**fields)
    read: dict[str, tuple[Document, list[Sheet]]] = {}
    out: list[dict[str, object]] = []
    for fid in fact_ids:
        checked = found.get(fid)
        if checked is None:
            out.append({"fact_id": fid, "error": f"not in {rows.name}"})
            continue
        if checked.document not in read:
            document = load(checked.document)
            read[checked.document] = (document, sheets(document))
        try:
            out.append(trace_json(trace(checked, *read[checked.document])))
        except ValueError as e:
            out.append({"fact_id": fid, "error": str(e)})
    return out


def newest_rows(directory: Path = CROSSCHECK_DIR) -> Path:
    """The newest cross-check run's rows; SystemExit when there is none."""
    runs = sorted(directory.glob("crosscheck-*.rows.jsonl"))
    if not runs:
        raise SystemExit(f"no cross-check run in {directory}: python -m aigent.facts crosscheck")
    return runs[-1]


def review_page(
    name: str,
    outcome: PageOutcome | None,
    rows_path: Path | None,
    add: int,
    url: str | None,
    lists: Path = REVIEW_PAGES,
    out_dir: Path = REVIEW_PAGES_OUT,
    load: Callable[[str], Document] = load_verify_document,
    image: Callable[[Path, int], str] = page_image,
    manifest: Mapping[str, Entry] | None = None,
) -> Path:
    """Builds review page `name`, drawing `add` new cards first and recording `url`; its card list
    is kept in `lists`, the page written under `out_dir`. Returns the page's path."""
    rows_path = rows_path or newest_rows()
    list_path = lists / f"{name}.json"
    page = read_page_list(list_path)
    if page is None:
        if outcome is None:
            raise SystemExit(f"{name} is a new page: give its --outcome, agree or differ")
        page = PageList(outcome, None, ())
    elif outcome is not None and outcome != page.outcome:
        raise SystemExit(f"{name} shows {page.outcome} facts, not {outcome}")
    rows = [Checked(**json.loads(line)) for line in rows_path.read_text().splitlines() if line]
    manifest = read_manifest() if manifest is None else manifest
    if add:
        kinds = kinds_of((r.document for r in rows), manifest)
        page = draw(rows, page, kinds, add, REVIEW_PAGE_SEED)
    if url:
        page = replace(page, url=url)
    write_page_list(list_path, page)
    ids = [c.fact_id for c in page.cards]
    traces = {cast(int, t["fact_id"]): t for t in trace_facts(rows_path, ids, load)}
    cards = page_cards(page, rows, traces, manifest, image=image)
    still_open = sum(r.outcome == page.outcome and r.tier == "xbrl-ai" for r in rows)
    html = build_page(cards, page.outcome, still_open, TEMPLATE.read_text(encoding="utf-8"))
    target = out_dir / name / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    batch = max((c.batch for c in page.cards), default=0)
    print(
        f"{len(cards)} cards ({add} new in batch {batch}) from {rows_path.name}, "
        f"{len(html.encode()) / 1e6:.1f} MB: {target}; publish it to {page.url or 'a new artifact'}"
    )
    return target


def tree_rss_gb(pid: int) -> float:
    """The resident memory of a process and everything it started, in GB; 0 once it has gone."""
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0.0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total / 2**30


def guarded(
    argv: Sequence[str],
    max_gb: float = VERIFY_REOCR_MAX_GB,
    poll_s: float = VERIFY_REOCR_POLL_S,
    rss_gb: Callable[[int], float] = tree_rss_gb,
) -> tuple[str, float]:
    """Runs `argv`, stopping it and all it started once their memory passes `max_gb`: its outcome
    ("ok", "failed exit N" or "stopped over N GB") and the peak seen."""
    child = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0.0
    while child.poll() is None:
        peak = max(peak, rss_gb(child.pid))
        if peak > max_gb:
            for process in psutil.Process(child.pid).children(recursive=True):
                process.kill()
            child.kill()
            child.wait()
            return f"stopped over {max_gb:g} GB", peak
        time.sleep(poll_s)
    return ("ok" if child.returncode == 0 else f"failed exit {child.returncode}"), peak


def reocr_queue(
    manifest: dict[str, Entry],
    documents: Sequence[str],
    tickers: Sequence[str],
    built: Callable[[str], bool] = reocr_built,
) -> tuple[list[str], list[str]]:
    """The re-OCR'd readings asked for, every scan's when neither `documents` nor `tickers` names
    any, still to build; and the names asked for that have no reading, not being such a scan."""
    readings = [r for n, e in sorted(manifest.items()) for r in readings_of(n, e) if is_reading(r)]
    chosen = [
        r
        for r in readings
        if (not documents and not tickers)
        or filed_of(r) in documents
        or ticker_of(filed_of(r)) in tickers
    ]
    unknown = sorted(set(documents) - {filed_of(r) for r in readings})
    return [r for r in chosen if not built(r)], unknown


def reocr(documents: Sequence[str], tickers: Sequence[str]) -> None:
    """Builds each re-OCR'd reading asked for that is not built, one process each, guarded."""
    todo, unknown = reocr_queue(read_manifest(), documents, tickers)
    if unknown:
        raise SystemExit(f"not scans with a re-OCR'd reading: {', '.join(unknown)}")
    print(f"{len(todo)} re-OCR'd readings to build", flush=True)
    for reading in todo:
        start = time.monotonic()
        outcome, peak = guarded([sys.executable, "-m", "aigent.facts", "reocr", "--one", reading])
        print(
            f"{reading}: {outcome}, {time.monotonic() - start:.0f}s, peak {peak:.1f} GB", flush=True
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
