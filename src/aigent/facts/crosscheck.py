"""The XBRL cross-check: each fact of a cross-checked company against what its annual report or
results document prints, matched by value at the table's scale, in a column stating the fact's
period, in a table for its basis. An agreement makes the fact `xbrl-filed`; anything else stays
`xbrl-ai` for a person."""

from __future__ import annotations

import bisect
import hashlib
import inspect
import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from aigent.facts import headers, labels, printed, vocab
from aigent.facts.cfg_facts import (
    CROSSCHECK_DIFFER_LABEL_MIN,
    CROSSCHECK_DISTINCT_DIGITS,
    CROSSCHECK_EXCLUSIVE,
    CROSSCHECK_FACT_ACTIONS,
    CROSSCHECK_FILED_FRAMEWORK,
    CROSSCHECK_FRAMEWORKS,
    CROSSCHECK_LABEL_MIN,
    CROSSCHECK_ROW_TURNS,
    CROSSCHECK_SCALES,
    CROSSCHECK_SIGN_FREE,
    CROSSCHECK_SPAN_ROWS,
    CROSSCHECK_SYNONYMS,
    VERIFY_OCR_TICKERS,
)
from aigent.facts.documents import (
    VERIFY_DIR,
    Entry,
    filed_of,
    load_verify_document,
    read_manifest,
    reading_fingerprint,
    reocr_built,
    usable_readings,
)
from aigent.facts.headers import (
    Column,
    basis_in,
    column_periods,
    document_scale,
    framework_in,
    months_of,
    scale_in,
)
from aigent.facts.labels import label_match, note, per_share_row, serial
from aigent.facts.printed import MAIN_LINE, figure_row, number
from aigent.facts.share_events import PER_SHARE
from aigent.facts.store import XBRL_TIERS, demote_derived
from aigent.retrieval.chunk import Document

Outcome = Literal["agree", "differ", "unmatched"]
Rule = Literal["label", "distinct", "per-share"]  # what let an equal figure count as the fact
RowKind = Literal["heading", "main line", "figures"]
_RANK: dict[Outcome, int] = {"unmatched": 0, "differ": 1, "agree": 2}


def fingerprint(parts: Iterable[str]) -> str:
    """A short hash of `parts` in order."""
    return hashlib.sha256(json.dumps(list(parts)).encode()).hexdigest()[:12]


# Everything that decides an outcome: this module, the header, label, figure and word readers, their
# settings, and the readings.
CROSSCHECK_FINGERPRINT = fingerprint(
    (
        Path(__file__).read_text(encoding="utf-8"),
        *(inspect.getsource(m) for m in (headers, labels, printed, vocab)),
        repr(
            (
                CROSSCHECK_DIFFER_LABEL_MIN,
                CROSSCHECK_DISTINCT_DIGITS,
                CROSSCHECK_EXCLUSIVE,
                CROSSCHECK_FACT_ACTIONS,
                CROSSCHECK_FILED_FRAMEWORK,
                CROSSCHECK_FRAMEWORKS,
                CROSSCHECK_LABEL_MIN,
                CROSSCHECK_ROW_TURNS,
                CROSSCHECK_SCALES,
                CROSSCHECK_SIGN_FREE.pattern,
                CROSSCHECK_SPAN_ROWS,
                CROSSCHECK_SYNONYMS,
                VERIFY_OCR_TICKERS,
                PER_SHARE,
            )
        ),
        reading_fingerprint(),
    )
)


@dataclass(frozen=True)
class Cell:
    """One printed number: where it is, the period of the column it belongs to, its row's words,
    and its value as printed."""

    row: int
    column: int
    period: Column | None
    label: str
    printed: str
    value: float
    places: int
    percent: bool
    heading: str = ""  # the last row above it with no figures, as "Earnings per equity share"


@dataclass(frozen=True)
class RowRead:
    """How one body row was read: its own words, the label its figures carry (its heading's for a
    block's subtotal), the heading it sits under and its block."""

    row: int
    kind: RowKind
    label: str
    named: str
    heading: str
    block: int


@dataclass(frozen=True)
class Sheet:
    """A results table read for matching: its basis, scale and accounting framework from the text
    around it, its page, each column's period, its numbers, how many leading rows were header,
    and each body row's reading."""

    index: int
    basis: str | None
    scale: float | None
    page: int
    columns: Mapping[int, Column]
    cells: tuple[Cell, ...]
    header: int = 0
    rows: tuple[RowRead, ...] = ()
    framework: str | None = None


@dataclass(frozen=True)
class Fact:
    """An XBRL fact to check, with every label it is known by."""

    fact_id: int
    ticker: str
    basis: str
    concept: str
    labels: tuple[str, ...]
    segment: str | None
    period_start: date | None
    period_end: date
    value: float
    unit: str
    decimals: int | None
    tier: str
    document: str


@dataclass(frozen=True)
class Checked:
    """A fact's outcome, and for an agreement or a difference, the cell it rests on: its table's
    index in the document, row and column, the label share, and for an agreement, the rule."""

    fact_id: int
    ticker: str
    basis: str
    concept: str
    segment: str | None
    period_start: str | None
    period_end: str
    value: float
    unit: str
    tier: str
    document: str
    outcome: Outcome
    page: int | None = None
    printed: str | None = None
    row_label: str | None = None
    sheet: int | None = None
    row: int | None = None
    column: int | None = None
    heading: str | None = None
    share: float | None = None
    rule: Rule | None = None
    veto: str | None = None  # not agreed: the veto that refused an equal cell, and its row


def sheets(document: Document) -> list[Sheet]:
    """Every table with numbers, read for matching. Basis, scale and accounting framework come from
    the text since the table before and the table's own header lines, else the table before's,
    else, for scale, the one the document states most."""
    out: list[Sheet] = []
    basis: str | None = None
    framework: str | None = None
    scale: float | None = document_scale(document.text)
    previous_end = 0
    before_columns: Mapping[int, Column] = {}
    before_heading = ""
    before_width, before_page = 0, 0
    for index, table in enumerate(document.tables):
        before = document.text[previous_end : table.start].lower()
        previous_end = table.end
        cells_by_row = [[number(c) for c in row] for row in table.rows]
        first = next((i for i, row in enumerate(table.rows) if figure_row(row)), None)
        if first is None:
            continue
        header = table.rows[:first]
        own = " ".join(" ".join(row) for row in header).lower()
        basis = basis_in(before + " " + own) or basis
        framework = framework_in(before + " " + own) or framework
        scale = scale_in(before[-300:] + " " + own) or scale
        columns: Mapping[int, Column] = column_periods(header)
        width = max(len(row) for row in table.rows)
        page = bisect.bisect_right(document.page_starts, table.start) or 1
        if not columns and width == before_width and page - before_page in (0, 1):
            columns = (
                before_columns  # a statement continued past a page break, its header left behind
            )
        continued = columns is before_columns and bool(columns)
        heading = (
            before_heading if continued else " ".join(c for c in header[-1] if c) if header else ""
        )
        copies = _spanning(table.rows)
        first_dated = min(columns, default=None)
        body: list[tuple[int, list[tuple[int, tuple[float, int, bool]]], str, str, int]] = []
        read: list[RowRead] = []
        block = 0
        for r in range(first, len(table.rows)):
            row = table.rows[r]
            numbers = [
                (j, parsed)
                for j, parsed in enumerate(cells_by_row[r])
                if parsed is not None
                and row[j].strip()
                and not serial(j, row[j])
                and not note(j, row[j], first_dated)
            ]
            taken = {j for j, _ in numbers}
            label = " ".join(
                c
                for j, c in enumerate(row)
                if c and j not in taken and not serial(j, c) and not note(j, c, first_dated)
            )
            if not numbers:
                heading, block = label or heading, block + 1
                read.append(RowRead(r, "heading", label, "", heading, block))
                continue
            kind: RowKind = "figures"
            # a numbered main line (IX, XI, D.) heads its own block
            printed = (c for j, c in enumerate(row) if c and (r, j) not in copies)
            if MAIN_LINE.match(" ".join(printed)):
                heading, block, kind = label, block + 1, "main line"
            body.append((r, numbers, label, heading, block))
            read.append(RowRead(r, kind, label, label, heading, block))
        closes = {
            block: i for i, (_, _, label, _, block) in reversed(list(enumerate(body))) if not label
        }
        cells: list[Cell] = []
        named_as: dict[int, str] = {}
        for i, (r, numbers, label, under, block) in enumerate(body):
            subtotal = i == closes.get(block)  # its heading's first unlabelled row
            named = label or (under if subtotal else "")
            named_as[r] = named
            for j, parsed, period in _aligned(numbers, columns):
                cells.append(
                    Cell(r, j, period, named, table.rows[r][j].strip(), *parsed, heading=under)
                )
        rows = tuple(replace(x, named=named_as[x.row]) if x.row in named_as else x for x in read)
        out.append(Sheet(index, basis, scale, page, columns, tuple(cells), first, rows, framework))
        before_columns, before_width, before_page, before_heading = columns, width, page, heading
    return out


def _spanning(rows: Sequence[Sequence[str]]) -> set[tuple[int, int]]:
    """The (row, column) cells copying a spanning cell's text: a run of `CROSSCHECK_SPAN_ROWS` or
    more rows printing the same text in one column; the run's first row keeps it."""
    copies: set[tuple[int, int]] = set()
    for j in range(max((len(row) for row in rows), default=0)):
        texts = [row[j].strip() if j < len(row) else "" for row in rows]
        start = 0
        while start < len(texts):
            end = start + 1
            while end < len(texts) and texts[end] == texts[start]:
                end += 1
            if end - start >= CROSSCHECK_SPAN_ROWS:
                copies.update((r, j) for r in range(start + 1, end))
            start = end
    return copies


def _aligned(
    numbers: Sequence[tuple[int, tuple[float, int, bool]]], columns: Mapping[int, Column]
) -> list[tuple[int, tuple[float, int, bool], Column | None]]:
    """Each number with its column's period: by order when the row has one number per dated
    column, since a row's cells can sit a column off its header; else by its own column."""
    dated = sorted(columns)
    if len(numbers) == len(dated):
        return [(j, parsed, columns[k]) for (j, parsed), k in zip(numbers, dated, strict=True)]
    return [(j, parsed, columns.get(j)) for j, parsed in numbers]


def agrees(fact: Fact, cell: Cell, scale: float | None) -> bool:
    """The printed figure, at the table's scale, equal to the fact to the last printed digit; in
    size only for a concept naming its direction (`CROSSCHECK_SIGN_FREE`: a payment filed positive,
    printed in brackets)."""
    if fact.unit == "pure":
        held = cell.value / 100 if cell.percent else cell.value
        allowed = 0.5 * 10**-cell.places / (100 if cell.percent else 1)
        return abs(held - fact.value) <= allowed + 1e-12
    if cell.percent:
        return False
    factor = 1.0 if fact.unit == "INRPerShare" else scale
    if factor is None:
        return False
    allowed = 0.5 * 10**-cell.places * factor  # to the printed digit; XBRL's decimals can be coarse
    if CROSSCHECK_SIGN_FREE.search(fact.concept):
        return abs(abs(cell.value * factor) - abs(fact.value)) <= allowed + 1e-9
    return abs(cell.value * factor - fact.value) <= allowed + 1e-9


def _distinct(printed: str) -> bool:
    digits = re.sub(r"\D", "", printed).lstrip("0")
    return len(digits) >= CROSSCHECK_DISTINCT_DIGITS


def _in_period(fact: Fact, column: Column | None, columns: Mapping[int, Column]) -> bool:
    if column is None or column.end != fact.period_end:
        return False
    if fact.period_start is None:
        return True
    want = months_of(fact.period_start, fact.period_end)
    if column.months is not None:
        return column.months == want
    return sum(c.end == fact.period_end and c.months in (None, want) for c in columns.values()) == 1


def check_fact(fact: Fact, read: Sequence[Sheet]) -> Checked:
    """`agree` on the best-labelled cell of the fact's basis and period equal to it, its row
    carrying its label or the figure distinctive alone, the first on a tie; else `differ` on the
    best-labelled cell; else `unmatched`. Not agreed, it names the veto that refused the
    best-labelled cell equal to it (`labels.VETOES`), as "rule: row label"."""
    base = Checked(
        fact.fact_id,
        fact.ticker,
        fact.basis,
        fact.concept,
        fact.segment,
        None if fact.period_start is None else fact.period_start.isoformat(),
        fact.period_end.isoformat(),
        fact.value,
        fact.unit,
        fact.tier,
        fact.document,
        "unmatched",
    )
    agreed: tuple[float, Sheet, Cell, Rule] | None = None
    best: tuple[float, Sheet, Cell] | None = None
    vetoed: tuple[float, str, Cell] | None = None
    for sheet in read:
        if sheet.basis not in (None, fact.basis):
            continue
        if sheet.framework not in (None, CROSSCHECK_FILED_FRAMEWORK):
            continue
        for cell in sheet.cells:
            if not _in_period(fact, cell.period, sheet.columns):
                continue
            match = label_match(fact.labels, fact.segment, cell.label)
            share = match.share
            if agrees(fact, cell, sheet.scale):
                rule = _rule(fact, cell, share)
                if rule is not None and (agreed is None or share > agreed[0]):
                    agreed = (share, sheet, cell, rule)
                if match.veto and match.raw >= CROSSCHECK_LABEL_MIN:
                    if vetoed is None or match.raw > vetoed[0]:
                        vetoed = (match.raw, match.veto, cell)
            if share >= CROSSCHECK_DIFFER_LABEL_MIN and (best is None or share > best[0]):
                best = (share, sheet, cell)
    if agreed is not None:
        return _with(base, "agree", agreed[1], agreed[2], agreed[0], agreed[3])
    if vetoed is not None:
        base = replace(base, veto=f"{vetoed[1]}: {vetoed[2].label}")
    if best is not None:
        return _with(base, "differ", best[1], best[2], best[0], None)
    return base


def _rule(fact: Fact, cell: Cell, share: float) -> Rule | None:
    """What lets an equal figure count as the fact, tried in this order; None for nothing."""
    if share >= CROSSCHECK_LABEL_MIN:
        return "label"
    if _distinct(cell.printed):
        return "distinct"
    if fact.unit.endswith(PER_SHARE) and per_share_row(fact.labels, f"{cell.heading} {cell.label}"):
        return "per-share"
    return None


def _with(
    base: Checked, outcome: Outcome, sheet: Sheet, cell: Cell, share: float, rule: Rule | None
) -> Checked:
    return replace(
        base,
        outcome=outcome,
        page=sheet.page,
        printed=cell.printed,
        row_label=cell.label,
        sheet=sheet.index,
        row=cell.row,
        column=cell.column,
        heading=cell.heading,
        share=round(share, 3),
        rule=rule,
    )


def documents_for(fact: Fact, filed: date) -> tuple[str, ...]:
    """The stems of the documents a fact is printed in, in the order they are tried: its year's
    annual report for a year's figure or a 31 March balance in a March filing, then its quarter's
    results."""
    results = f"{fact.ticker}-{filed.isoformat()}"
    year_end = filed.month == 3 and fact.period_end == filed
    if year_end and months_of(fact.period_start, fact.period_end) in (0, 12):
        return (f"{fact.ticker}-FY{filed.year % 100:02d}", results)
    return (results,)


def facts_to_check(
    conn: sqlite3.Connection,
    tickers: Sequence[str],
    manifest: Mapping[str, Entry],
    ocr_tickers: Sequence[str] = VERIFY_OCR_TICKERS,
    built: Callable[[str], bool] = reocr_built,
) -> list[Fact]:
    """Every XBRL fact of `tickers` once per readable document it is printed in (`documents_for`,
    in that order), and after a scan once per re-OCR'd reading `built` holds, with its standard
    label and aliases; a fact with none is left out."""
    aliases: dict[str, list[str]] = {}
    for concept, label in conn.execute("SELECT concept, label FROM concept_labels"):
        aliases.setdefault(concept, []).append(label)
    documents: dict[str, list[str]] = {}
    for reading in usable_readings(manifest, ocr_tickers, built):
        documents.setdefault(Path(filed_of(reading)).stem, []).append(reading)
    marks = ", ".join("?" for _ in tickers)
    tiers = ", ".join("?" for _ in XBRL_TIERS)
    rows = conn.execute(
        f"""SELECT f.id, f.ticker, f.basis, f.concept, f.label, f.segment, f.period_start,
                   f.period_end, f.value, f.unit, f.decimals, f.tier, fl.period_end
            FROM facts f JOIN filings fl ON fl.id = f.filing_id
            WHERE f.ticker IN ({marks}) AND f.tier IN ({tiers})
            ORDER BY f.ticker, fl.period_end, f.basis, f.concept, f.segment, f.period_end""",
        (*tickers, *XBRL_TIERS),
    ).fetchall()
    out: list[Fact] = []
    for (
        fid,
        ticker,
        basis,
        concept,
        label,
        segment,
        start,
        end,
        value,
        unit,
        dec,
        tier,
        filed,
    ) in rows:
        fact = Fact(
            fid,
            ticker,
            basis,
            concept,
            (label, *aliases.get(concept, [])),
            segment,
            None if start is None else date.fromisoformat(start),
            date.fromisoformat(end),
            float(value),
            unit,
            dec,
            tier,
            "",
        )
        for stem in documents_for(fact, date.fromisoformat(filed)):
            out.extend(replace(fact, document=name) for name in documents.get(stem, ()))
    return out


def crosscheck(
    conn: sqlite3.Connection,
    tickers: Sequence[str],
    verify_dir: Path = VERIFY_DIR,
    load: Callable[[str, Path], Document] = load_verify_document,
    built: Callable[[str], bool] = reocr_built,
    ocr_tickers: Sequence[str] = VERIFY_OCR_TICKERS,
) -> list[Checked]:
    """Every fact of `tickers` checked against the documents it is printed in, in order until one
    agrees, keeping its best outcome; each document read once."""
    facts = facts_to_check(conn, tickers, read_manifest(verify_dir), ocr_tickers, built)
    read: dict[str, list[Sheet]] = {}
    best: dict[int, Checked] = {}
    for fact in facts:
        held = best.get(fact.fact_id)
        if held is not None and held.outcome == "agree":
            continue
        if fact.document not in read:
            read[fact.document] = sheets(load(fact.document, verify_dir))
        checked = check_fact(fact, read[fact.document])
        if held is None or _RANK[checked.outcome] > _RANK[held.outcome]:
            best[fact.fact_id] = checked
    return list(best.values())


def apply(
    conn: sqlite3.Connection,
    checked: Iterable[Checked],
    now: datetime,
    origins: Mapping[str, str],
    reading: str = CROSSCHECK_FINGERPRINT,
) -> int:
    """Makes each agreeing `xbrl-ai` fact `xbrl-filed`, naming its document as `<origin>:<path>`,
    keeping the printed figure and page, and stamping the `reading` it agreed by; a fact a person
    checked, or already agreed, is left alone. Returns how many; a document with no origin raises
    KeyError and writes nothing."""
    stamp = now.isoformat(timespec="seconds")
    moved = 0
    with conn:
        for c in checked:
            if c.outcome != "agree" or c.tier != "xbrl-ai":
                continue
            cursor = conn.execute(
                """UPDATE facts SET tier = 'xbrl-filed', verified_by = ?, verified_at = ?,
                          printed = ?, page = ?, verified_with = ?
                   WHERE id = ? AND tier = 'xbrl-ai'""",
                (
                    f"{origins[c.document]}:{c.document}",
                    stamp,
                    c.printed,
                    c.page,
                    reading,
                    c.fact_id,
                ),
            )
            moved += cursor.rowcount
    return moved


@dataclass(frozen=True)
class Rechecked:
    """What re-checking stale stamps did: agreements stamped afresh, facts made `xbrl-ai` again,
    and derived facts made `derived-ai` for losing a checked input."""

    restamped: int
    demoted: int
    derived_demoted: int


def recheck(
    conn: sqlite3.Connection,
    checked: Iterable[Checked],
    now: datetime,
    origins: Mapping[str, str],
    tickers: Sequence[str],
    reading: str = CROSSCHECK_FINGERPRINT,
) -> Rechecked:
    """Each `xbrl-filed` fact of `tickers` stamped by another reading than `reading`: stamped
    afresh when it agrees in `checked`, else made `xbrl-ai` again, with every derived fact that
    loses a checked input. A document with no origin raises KeyError and writes nothing."""
    by_id = {c.fact_id: c for c in checked}
    marks = ", ".join("?" for _ in tickers)
    stale = [
        int(r[0])
        for r in conn.execute(
            f"""SELECT id FROM facts WHERE tier = 'xbrl-filed' AND ticker IN ({marks})
                  AND verified_with IS NOT ?""",
            (*tickers, reading),
        )
    ]
    stamp = now.isoformat(timespec="seconds")
    restamped = demoted = 0
    with conn:
        for fid in stale:
            c = by_id.get(fid)
            if c is not None and c.outcome == "agree":
                conn.execute(
                    """UPDATE facts SET verified_by = ?, verified_at = ?, printed = ?, page = ?,
                              verified_with = ?
                       WHERE id = ?""",
                    (f"{origins[c.document]}:{c.document}", stamp, c.printed, c.page, reading, fid),
                )
                restamped += 1
            else:
                conn.execute(
                    """UPDATE facts SET tier = 'xbrl-ai', verified_by = NULL, verified_at = NULL,
                              printed = NULL, page = NULL, verified_with = NULL
                       WHERE id = ?""",
                    (fid,),
                )
                demoted += 1
        derived = demote_derived(conn)
    return Rechecked(restamped, demoted, derived)
