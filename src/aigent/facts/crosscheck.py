"""The XBRL cross-check: each fact of a cross-checked company against what its results document
prints, matched by value at the table's scale, in a column stating the fact's period, in a table for
its basis. An agreement makes the fact `xbrl-pdf`; anything else stays `xbrl-ai` for a person."""

from __future__ import annotations

import bisect
import json
import random
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from aigent.facts.cfg_facts import (
    CROSSCHECK_DIFFER_LABEL_MIN,
    CROSSCHECK_DISTINCT_DIGITS,
    CROSSCHECK_EXCLUSIVE,
    CROSSCHECK_LABEL_MIN,
    CROSSCHECK_SCALES,
    CROSSCHECK_SYNONYMS,
)
from aigent.facts.documents import VERIFY_DIR, load_verify_document, read_manifest
from aigent.facts.store import XBRL_TIERS
from aigent.retrieval.chunk import Document

Outcome = Literal["agree", "differ", "unmatched"]

_ZERO = {"..", "-", "–", "—", "nil", "0"}
_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        1,
    )
    for m in names
}
_NUMERIC_DATE = re.compile(r"(\d{1,2})[.,/-](\d{1,2})[.,/-](\d{4})")
_WORDED_DATE = re.compile(r"([a-z]{3,9})\.?\s*(\d{1,2})\s*,?\s*(\d{4})")
_SCALE = re.compile(r"\bin\s*(crore|lac|lakh|million)s?\b")
_STOP = {"of", "and", "the", "in", "on", "or", "to", "for", "from", "by", "a", "an", "with", "at"}


@dataclass(frozen=True)
class Column:
    """A table column's period as its header states it: the day it ends, and its length in months,
    0 for a balance at that day, None when the header does not say."""

    end: date | None
    months: int | None


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


@dataclass(frozen=True)
class Sheet:
    """A results table read for matching: its basis and scale from the text around it, its page,
    each column's period, and its numbers."""

    index: int
    basis: str | None
    scale: float | None
    page: int
    columns: Mapping[int, Column]
    cells: tuple[Cell, ...]


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
    """A fact's outcome, and for an agreement or a difference, the cell it rests on."""

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


def number(text: str) -> tuple[float, int, bool] | None:
    """A cell as a printed number: its value, decimal places and whether it is a percentage; None
    for text, a date, or two numbers in one cell. Dashes and `..` print a zero."""
    t = text.strip().replace("−", "-")
    if t.lower() in _ZERO:
        return 0.0, 0, False
    if _NUMERIC_DATE.search(t) or _WORDED_DATE.search(t.lower()):
        return None
    t = re.sub(r"^[.\s~]+(?=[\d(])", "", t)  # a leader of dots before the figure
    if re.fullmatch(r"[\d,]+\.\s+\d+", t):  # a stray space after the point
        t = re.sub(r"\s+", "", t)
    match = re.fullmatch(r"(\()?(-)?([\d,.]+)(\))?\s*(%)?", t)
    if not match or not re.search(r"\d", match.group(3)):
        return None
    digits = match.group(3)
    if digits.count(".") == 0 and re.search(r",\d{2}$", digits):  # a decimal comma
        digits = digits[: digits.rfind(",")] + "." + digits[digits.rfind(",") + 1 :]
    if digits.count(".") > 1:  # points misread for separators
        head, _, tail = digits.rpartition(".")
        digits = head.replace(".", "") + "." + tail
    digits = digits.replace(",", "")
    if not re.fullmatch(r"\d+(\.\d+)?", digits):
        return None
    places = len(digits.partition(".")[2])
    value = float(digits)
    if match.group(1) or match.group(2):  # a closing bracket can fall in the next cell
        value = -value
    return value, places, bool(match.group(5))


def _date_in(text: str) -> date | None:
    found: list[date] = []
    for d, m, y in _NUMERIC_DATE.findall(text):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            continue
    for name, d, y in _WORDED_DATE.findall(text.lower()):
        month = _MONTHS.get(name)
        if month is None:
            continue
        try:
            found.append(date(int(y), month, int(d)))
        except ValueError:
            continue
    return found[-1] if found else None


def _months_in(text: str) -> int | None:
    t = text.lower()
    if re.search(r"nine months|\b9m\b|9 months", t):
        return 9
    if re.search(r"six months|half[ -]?year|\bh1\b|6 months", t):
        return 6
    if re.search(r"three months|quarter|\bq[1-4l]\b|\bq[1-4l]-", t):
        return 3
    if re.search(r"\byear\b|\bfy", t):
        return 12
    if re.search(r"\bas\s*at\b|\basat\b|^\s*at\b", t):
        return 0
    return None


def column_periods(header: Sequence[Sequence[str]]) -> dict[int, Column]:
    """Each column's period from the header rows. A group heading covers the empty cells to its
    right ("Quarter ended" over three dated columns), but a column's own cells ("Q2-2026") say
    its length first."""
    width = max((len(row) for row in header), default=0)
    own: dict[int, list[str]] = {j: [] for j in range(width)}
    carried: dict[int, list[str]] = {j: [] for j in range(width)}
    for row in header:
        carry = ""
        for j in range(width):
            cell = row[j].strip() if j < len(row) else ""
            if cell:
                carry = cell
                own[j].append(cell)
            elif carry:
                carried[j].append(carry)
    columns: dict[int, Column] = {}
    for j in range(width):
        end = _date_in(" ".join(own[j])) or _date_in(" ".join(carried[j]))
        if end is not None:
            months = _months_in(" ".join(own[j]))
            columns[j] = Column(end, _months_in(" ".join(carried[j])) if months is None else months)
    return columns


def _scale_in(text: str) -> float | None:
    """The last scale the text states as "in crore", "in lac" and the like."""
    found = _SCALE.findall(text.lower())
    return CROSSCHECK_SCALES[found[-1]] if found else None


def document_scale(text: str) -> float | None:
    """The scale a document states most often as "in crore", "in lac" and the like."""
    found = _SCALE.findall(text.lower())
    if not found:
        return None
    return CROSSCHECK_SCALES[max(set(found), key=found.count)]


def _basis_in(text: str) -> str | None:
    found = [(m.start(), m.group(0)) for m in re.finditer(r"standalone|consolidated", text)]
    return max(found)[1] if found else None


def sheets(document: Document) -> list[Sheet]:
    """Every table with numbers, read for matching. Basis and scale come from the text since the
    table before and the table's own header lines, else the table before's, else, for scale, the
    one the document states most."""
    out: list[Sheet] = []
    basis: str | None = None
    scale: float | None = document_scale(document.text)
    previous_end = 0
    for index, table in enumerate(document.tables):
        before = document.text[previous_end : table.start].lower()
        previous_end = table.end
        cells_by_row = [[number(c) for c in row] for row in table.rows]
        first = next(
            (i for i, parsed in enumerate(cells_by_row) if sum(p is not None for p in parsed) >= 2),
            None,
        )
        if first is None:
            continue
        header = table.rows[:first]
        own = " ".join(" ".join(row) for row in header).lower()
        basis = _basis_in(before + " " + own) or basis
        scale = _scale_in(before[-300:] + " " + own) or scale
        columns = column_periods(header)
        cells: list[Cell] = []
        for r in range(first, len(table.rows)):
            row = table.rows[r]
            numbers = [
                (j, parsed)
                for j, parsed in enumerate(cells_by_row[r])
                if parsed is not None and row[j].strip() and not _serial(j, row[j])
            ]
            taken = {j for j, _ in numbers}
            label = " ".join(
                c for j, c in enumerate(row) if c and j not in taken and not _serial(j, c)
            )
            for j, parsed, period in _aligned(numbers, columns):
                cells.append(Cell(r, j, period, label, row[j].strip(), *parsed))
        page = bisect.bisect_right(document.page_starts, table.start) or 1
        out.append(Sheet(index, basis, scale, page, columns, tuple(cells)))
    return out


def _serial(column: int, text: str) -> bool:
    """A row's serial number in its first cell ("1", "12"), not a figure."""
    return column == 0 and re.fullmatch(r"\d{1,2}", text.strip()) is not None


def _aligned(
    numbers: Sequence[tuple[int, tuple[float, int, bool]]], columns: Mapping[int, Column]
) -> list[tuple[int, tuple[float, int, bool], Column | None]]:
    """Each number with its column's period: by order when the row has one number per dated
    column, since a row's cells can sit a column off its header; else by its own column."""
    dated = sorted(columns)
    if len(numbers) == len(dated):
        return [(j, parsed, columns[k]) for (j, parsed), k in zip(numbers, dated, strict=True)]
    return [(j, parsed, columns.get(j)) for j, parsed in numbers]


def months_of(start: date | None, end: date) -> int:
    if start is None:
        return 0
    return round(((end - start).days + 1) / 30.44)


def _words(text: str) -> set[str]:
    """A label's words, abbreviations read in full, little words dropped."""
    found: list[str] = re.findall(r"[a-z]+", text.lower())
    spelled = " ".join(CROSSCHECK_SYNONYMS.get(word, word) for word in found)
    return set(spelled.split()) - _STOP


def label_share(labels: Iterable[str], segment: str | None, row_label: str) -> float:
    """The largest share of any of a fact's labels' words, its segment's added, found in the row;
    0 for a row naming only the other of an exclusive pair the label names ("basic", "diluted")."""
    row = _words(row_label)
    best = 0.0
    for label in labels:
        words = _words(f"{label} {segment or ''}")
        if not words or any(
            mine in words and other in row and mine not in row
            for pair in CROSSCHECK_EXCLUSIVE
            for mine, other in (pair, pair[::-1])
        ):
            continue
        best = max(best, len(words & row) / len(words))
    return best


def agrees(fact: Fact, cell: Cell, scale: float | None) -> bool:
    """The printed figure, at the table's scale, equal to the fact to the last printed digit."""
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
    return abs(cell.value * factor - fact.value) <= allowed + 1e-9


def _distinct(printed: str) -> bool:
    digits = re.sub(r"\D", "", printed).lstrip("0")
    return len(digits) >= CROSSCHECK_DISTINCT_DIGITS


def _in_period(fact: Fact, column: Column | None, columns: Mapping[int, Column]) -> bool:
    if column is None or column.end != fact.period_end:
        return False
    if fact.period_start is None:
        return True
    if column.months is not None:
        return column.months == months_of(fact.period_start, fact.period_end)
    return sum(c.end == fact.period_end for c in columns.values()) == 1


def check_fact(fact: Fact, read: Sequence[Sheet]) -> Checked:
    """`agree` on a cell of the fact's basis and period equal to it, its row carrying its label
    or the figure distinctive alone; else `differ` on the best-labelled cell; else `unmatched`."""
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
    best: tuple[float, Sheet, Cell] | None = None
    for sheet in read:
        if sheet.basis not in (None, fact.basis):
            continue
        for cell in sheet.cells:
            if not _in_period(fact, cell.period, sheet.columns):
                continue
            share = label_share(fact.labels, fact.segment, cell.label)
            if agrees(fact, cell, sheet.scale) and (
                share >= CROSSCHECK_LABEL_MIN or _distinct(cell.printed)
            ):
                return _with(base, "agree", sheet, cell)
            if share >= CROSSCHECK_DIFFER_LABEL_MIN and (best is None or share > best[0]):
                best = (share, sheet, cell)
    if best is not None:
        return _with(base, "differ", best[1], best[2])
    return base


def _with(base: Checked, outcome: Outcome, sheet: Sheet, cell: Cell) -> Checked:
    return replace(
        base, outcome=outcome, page=sheet.page, printed=cell.printed, row_label=cell.label
    )


def facts_to_check(
    conn: sqlite3.Connection, tickers: Sequence[str], manifest: Mapping[str, object]
) -> list[Fact]:
    """Every XBRL fact of `tickers` whose quarter has a results document in the manifest, with its
    standard label and aliases."""
    aliases: dict[str, list[str]] = {}
    for concept, label in conn.execute("SELECT concept, label FROM concept_labels"):
        aliases.setdefault(concept, []).append(label)
    documents = {Path(name).stem: name for name in manifest if name.startswith("results/")}
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
        document = documents.get(f"{ticker}-{filed}")
        if document is None:
            continue
        out.append(
            Fact(
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
                document,
            )
        )
    return out


def crosscheck(
    conn: sqlite3.Connection,
    tickers: Sequence[str],
    verify_dir: Path = VERIFY_DIR,
    load: Callable[[str, Path], Document] = load_verify_document,
) -> list[Checked]:
    """Every fact of `tickers` checked against its quarter's results document, each document read
    once."""
    facts = facts_to_check(conn, tickers, read_manifest(verify_dir))
    read: dict[str, list[Sheet]] = {}
    out: list[Checked] = []
    for fact in facts:
        if fact.document not in read:
            read[fact.document] = sheets(load(fact.document, verify_dir))
        out.append(check_fact(fact, read[fact.document]))
    return out


def apply(conn: sqlite3.Connection, checked: Iterable[Checked], now: datetime) -> int:
    """Makes each agreeing `xbrl-ai` fact `xbrl-pdf`, naming its document and keeping the printed
    figure and page; a fact a person checked, or already agreed, is left alone. Returns how many."""
    stamp = now.isoformat(timespec="seconds")
    moved = 0
    with conn:
        for c in checked:
            if c.outcome != "agree" or c.tier != "xbrl-ai":
                continue
            cursor = conn.execute(
                """UPDATE facts SET tier = 'xbrl-pdf', verified_by = ?, verified_at = ?,
                          printed = ?, page = ?
                   WHERE id = ? AND tier = 'xbrl-ai'""",
                (f"pdf:{c.document}", stamp, c.printed, c.page, c.fact_id),
            )
            moved += cursor.rowcount
    return moved


def review_sample(checked: Sequence[Checked], size: int, seed: int) -> list[Checked]:
    """A seeded random sample of the agreements, for a person to check."""
    agreed = [c for c in checked if c.outcome == "agree"]
    return sorted(
        random.Random(seed).sample(agreed, min(size, len(agreed))), key=lambda c: c.fact_id
    )


def write_report(
    checked: Sequence[Checked], sample: Sequence[Checked], header: Mapping[str, str], path: Path
) -> Path:
    """The counts per company, basis and outcome, then the review queue: every fact not agreed,
    and the sampled agreements; every row beside it as JSON lines."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for c in checked:
        counts.setdefault((c.ticker, c.basis), {"agree": 0, "differ": 0, "unmatched": 0})
        counts[(c.ticker, c.basis)][c.outcome] += 1
    lines = ["# XBRL cross-check", ""]
    lines += [f"- **{k}:** {v}" for k, v in header.items()]
    lines += ["", "| company | basis | agree | differ | unmatched |", "|---|---|---|---|---|"]
    for (ticker, basis), n in sorted(counts.items()):
        lines.append(f"| {ticker} | {basis} | {n['agree']} | {n['differ']} | {n['unmatched']} |")
    lines += ["", "## Review queue: not agreed", ""]
    lines += ["| fact | concept | segment | period | XBRL | printed | outcome | document p. |"]
    lines += ["|---|---|---|---|---|---|---|---|"]
    for c in checked:
        if c.outcome != "agree":
            lines.append(_row(c))
    lines += ["", "## Review queue: sampled agreements", ""]
    lines += ["| fact | concept | segment | period | XBRL | printed | outcome | document p. |"]
    lines += ["|---|---|---|---|---|---|---|---|"]
    lines += [_row(c) for c in sample]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = path.with_suffix(".rows.jsonl")
    rows.write_text("".join(json.dumps(asdict(c)) + "\n" for c in checked), encoding="utf-8")
    return path


def _row(c: Checked) -> str:
    period = c.period_end if c.period_start is None else f"{c.period_start} to {c.period_end}"
    where = f"{c.document} p.{c.page}" if c.page else c.document
    return (
        f"| {c.fact_id} | {c.concept} | {c.segment or ''} | {period} | {c.value:g} {c.unit} | "
        f"{c.printed or ''} | {c.outcome} | {where} |"
    )
