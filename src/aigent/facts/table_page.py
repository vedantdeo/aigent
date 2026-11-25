"""Table pages: tables of the filed documents drawn as cards for a person to tag what is wrong with
how each was read, each with its printed page, its rows as the cross-check reads them and as Docling
parsed them, and audit flags. A page's table list is kept in
`evals/reference/table-pages/<name>.json`."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

from aigent.facts.cfg_facts import (
    REVIEW_PAGE_KINDS,
    REVIEW_PAGE_MAX_MB,
    TABLE_PAGE_FLAGGED_SHARE,
    TABLE_PAGE_MAX_ROWS,
    TABLE_PAGE_TAGS,
)
from aigent.facts.crosscheck import Sheet
from aigent.facts.headers import Column
from aigent.facts.printed import number
from aigent.facts.review_page import shares
from aigent.facts.trace import period_text
from aigent.retrieval.chunk import Table

TABLE_PAGES = Path(__file__).resolve().parents[3] / "evals" / "reference" / "table-pages"
TEMPLATE = Path(__file__).with_name("table_page.html")
FLAGS: tuple[str, ...] = (
    "merged figures",
    "text in a figure column",
    "columns share a period",
    "no period read",
    "total not footing",
)
_MARKERS = ("{{TITLE}}", "{{LEDE}}", "{{TAGS}}", "{{DATA}}")
_FIGURE = re.compile(r"[(\[{]?-?[\d,]+(?:\.\d+)?[)\]}]?\.?")  # one printed figure, as a word
_WORD = re.compile(r"[A-Za-z]{3,}")
_FOOT_SPAN = 15  # rows above a total its figures may sum


def is_figure(word: str) -> bool:
    """A word printing a figure of two or more digits, as `1,234.5` or `(12)`."""
    return bool(_FIGURE.fullmatch(word)) and len(re.sub(r"\D", "", word)) >= 2


def figures_in(rows: Iterable[Sequence[str]]) -> Counter[str]:
    """Every figure printed in `rows`, brackets dropped, counted."""
    return Counter(w.strip("[]{}") for row in rows for c in row for w in c.split() if is_figure(w))


def _label(row: Sequence[str]) -> str:
    return " ".join(c for c in row if c and number(c) is None)


def unfooted(rows: Sequence[Sequence[str]], header: int) -> list[tuple[int, int]]:
    """Each (row, column) of a body row naming a total whose figure is not the sum of two to
    `_FOOT_SPAN` figures just above it in its column."""
    out: list[tuple[int, int]] = []
    for i in range(header, len(rows)):
        if "total" not in _label(rows[i]).lower():
            continue
        for j, cell in enumerate(rows[i]):
            read = number(cell)
            if read is None or read[0] == 0:
                continue
            total, seen, foots = 0.0, 0, False
            for k in range(i - 1, max(header - 1, i - _FOOT_SPAN - 1), -1):
                above = number(rows[k][j]) if j < len(rows[k]) else None
                if above is None:
                    continue
                total, seen = total + above[0], seen + 1
                if seen >= 2 and abs(total - read[0]) <= max(0.011, abs(read[0]) * 1e-6):
                    foots = True
                    break
            if not foots:
                out.append((i, j))
    return out


@dataclass(frozen=True)
class Audit:
    """What the audits found in one table: each flag's cells as (row, column), or columns for the
    period flags."""

    merged: tuple[tuple[int, int], ...]
    spilled: tuple[tuple[int, int], ...]
    shared: tuple[int, ...]
    unread: tuple[int, ...]
    unfooted: tuple[tuple[int, int], ...]

    def counts(self) -> dict[str, int]:
        """Each flag's count, in `FLAGS` order."""
        found = (self.merged, self.spilled, self.shared, self.unread, self.unfooted)
        return {flag: len(f) for flag, f in zip(FLAGS, found, strict=True)}

    def flagged(self) -> bool:
        return any(self.counts().values())


def audit(table: Table, sheet: Sheet) -> Audit:
    """A read table's flags: body cells holding two or more figures, or a figure and words, in a
    column with figures or a period; three or more figure columns given one period; figure
    columns given none; and totals that do not foot."""
    rows = table.rows
    figure_columns = sorted({c.column for c in sheet.cells} | set(sheet.columns))
    merged: list[tuple[int, int]] = []
    spilled: list[tuple[int, int]] = []
    for i in range(sheet.header, len(rows)):
        for j in figure_columns:
            cell = rows[i][j] if j < len(rows[i]) else ""
            words = cell.split()
            figures = sum(is_figure(w) for w in words)
            if figures >= 2:
                merged.append((i, j))
            elif figures and any(_WORD.search(w) and not is_figure(w) for w in words):
                spilled.append((i, j))
    periods: Counter[Column] = Counter(
        sheet.columns[j] for j in figure_columns if j in sheet.columns
    )
    shared = tuple(
        j for j in figure_columns if j in sheet.columns and periods[sheet.columns[j]] >= 3
    )
    unread = tuple(
        j for j in figure_columns if j not in sheet.columns or sheet.columns[j].end is None
    )
    return Audit(tuple(merged), tuple(spilled), shared, unread, tuple(unfooted(rows, sheet.header)))


@dataclass(frozen=True)
class Found:
    """A table a page can draw: its document, its index there, its kind of copy and whether an
    audit flags it."""

    document: str
    sheet: int
    kind: str
    flagged: bool


@dataclass(frozen=True)
class KeptTable:
    """A table on a page: its card id, document, index there, batch and the pool it was drawn
    from, `flagged` or `any`, or `differs` for one chosen by its facts that differ."""

    id: str
    document: str
    sheet: int
    batch: int
    pool: str


@dataclass(frozen=True)
class TableList:
    """A table page's tables and the artifact it is published at."""

    url: str | None
    cards: tuple[KeptTable, ...]


class _Stored(TypedDict):
    url: str | None
    cards: list[dict[str, object]]


def read_table_list(path: Path) -> TableList:
    """A page's table list; an empty one when it has none yet."""
    if not path.exists():
        return TableList(None, ())
    stored = cast(_Stored, json.loads(path.read_text(encoding="utf-8")))
    cards = tuple(
        KeptTable(
            str(c["id"]),
            str(c["document"]),
            int(cast(int, c["sheet"])),
            int(cast(int, c["batch"])),
            str(c["pool"]),
        )
        for c in stored["cards"]
    )
    return TableList(stored["url"], cards)


def write_table_list(path: Path, page: TableList) -> None:
    """Writes a page's table list sorted by card id, one card a line, its fields in one order."""
    head = json.dumps({"url": page.url}, ensure_ascii=False)
    lines = ",\n".join(
        "  " + json.dumps(asdict(c), ensure_ascii=False)
        for c in sorted(page.cards, key=lambda c: c.id)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{head[:-1]}, "cards": [\n{lines}\n]}}\n', encoding="utf-8")


def draw_tables(
    found: Sequence[Found],
    page: TableList,
    size: int,
    seed: int,
    flagged_share: float = TABLE_PAGE_FLAGGED_SHARE,
    order: Sequence[str] = REVIEW_PAGE_KINDS,
) -> TableList:
    """`page` with a new batch of `size` tables not already on it: `flagged_share` of them from
    those an audit flags, the rest from all, each part spread evenly over the kinds of copy."""
    taken = {(c.document, c.sheet) for c in page.cards}
    batch = max((c.batch for c in page.cards), default=0) + 1
    rng = random.Random(f"{seed}:{batch}")
    drawn: list[tuple[Found, str]] = []
    for pool in ("flagged", "any"):
        want = round(size * flagged_share) if pool == "flagged" else size - len(drawn)
        by_kind: dict[str, list[Found]] = {}
        for f in found:
            if (f.document, f.sheet) not in taken and (pool == "any" or f.flagged):
                by_kind.setdefault(f.kind, []).append(f)
        split = shares(want, {k: len(v) for k, v in by_kind.items()}, order)
        for kind in order:
            for f in rng.sample(by_kind.get(kind, []), split[kind]):
                drawn.append((f, pool))
                taken.add((f.document, f.sheet))
    rng.shuffle(drawn)
    start = max((int(c.id[1:]) for c in page.cards), default=0) + 1
    new = tuple(
        KeptTable(f"t{n:02d}", f.document, f.sheet, batch, pool)
        for n, (f, pool) in enumerate(drawn, start=start)
    )
    return TableList(page.url, page.cards + new)


@dataclass(frozen=True)
class CardSource:
    """What a card shows beyond its table: its kind and ticker, Docling's own rows when the table
    was re-rowed, the filed copy's URL and the page image."""

    kind: str
    ticker: str
    docling: tuple[tuple[str, ...], ...] | None
    url: str | None
    image: str | None


def table_card(
    kept: KeptTable,
    table: Table,
    sheet: Sheet,
    source: CardSource,
    max_rows: int = TABLE_PAGE_MAX_ROWS,
) -> dict[str, object]:
    """A table's card data: its rows as read, each with its reading and the cells an audit flags,
    Docling's rows when they differ, its column periods, flags, and the figures Docling had that
    the reading lost."""
    found = audit(table, sheet)
    marked: dict[int, dict[str, list[int]]] = {}
    for name, cells in (
        ("merged", found.merged),
        ("spilled", found.spilled),
        ("unfooted", found.unfooted),
    ):
        for i, j in cells:
            marked.setdefault(i, {}).setdefault(name, []).append(j)
    by_row = {r.row: r for r in sheet.rows}
    rows = [
        {
            "row": i,
            "cells": list(cells),
            "docling_header": i < table.header_rows,
            "section": i in table.sections,
            "read": None if i not in by_row else asdict(by_row[i]),
            "marks": marked.get(i, {}),
        }
        for i, cells in enumerate(table.rows[:max_rows])
    ]
    docling = None if source.docling is None or source.docling == table.rows else source.docling
    lost = figures_in(docling) - figures_in(table.rows) if docling is not None else Counter()
    return {
        "id": kept.id,
        "batch": kept.batch,
        "pool": kept.pool,
        "kind": source.kind,
        "ticker": source.ticker,
        "document": kept.document,
        "sheet": kept.sheet,
        "page": sheet.page,
        "basis": sheet.basis,
        "scale": sheet.scale,
        "header": sheet.header,
        "columns": {j: period_text(c) for j, c in sorted(sheet.columns.items())},
        "shared": list(found.shared),
        "unread": list(found.unread),
        "flags": found.counts(),
        "rows": rows,
        "more_rows": max(0, len(table.rows) - max_rows),
        "docling": None if docling is None else [list(r) for r in docling[:max_rows]],
        "lost": sorted(lost.elements())[:40],
        "lost_count": sum(lost.values()),
        "url": source.url,
        "image": source.image,
    }


def _data(cards: Sequence[Mapping[str, object]]) -> str:
    """The cards as JSON for a script tag: a `</` escaped, and OCR's U+FFFD kept as an escape."""
    return (
        json.dumps(list(cards), ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\ufffd", "\\ufffd")
    )


def build_table_page(
    cards: Sequence[Mapping[str, object]],
    template: str,
    tags: Sequence[str] = TABLE_PAGE_TAGS,
    max_mb: float = REVIEW_PAGE_MAX_MB,
) -> str:
    """The page: `template` with its title, lede (saying how its cards were chosen, by their
    pool), tags and cards filled in. ValueError for a template missing a marker, or a page over
    `max_mb`."""
    missing = [m for m in _MARKERS if m not in template]
    if missing:
        raise ValueError(f"the template lacks {', '.join(missing)}")
    pools = {str(c.get("pool")) for c in cards}
    chosen = (
        "the tables the most cross-checked facts differ in, most first"
        if pools == {"differs"}
        else "drawn in batches of half flagged by an audit and half at random, spread across the "
        "kinds of copy"
    )
    lede = (
        f"{len(cards)} tables of the filed documents, {chosen}. For each, compare the table as "
        f"read with the printed page, tag what is wrong and say where in a note."
    )
    page = (
        template.replace("{{TITLE}}", "Table Check")
        .replace("{{LEDE}}", lede)
        .replace("{{TAGS}}", json.dumps(list(tags), ensure_ascii=False))
        .replace("{{DATA}}", _data(cards))
    )
    size = len(page.encode()) / 1e6
    if size > max_mb:
        raise ValueError(f"the page is {size:.1f} MB, over {max_mb} MB: draw fewer tables")
    return page


def found_tables(
    readings: Iterable[str],
    load: Callable[[str], tuple[Sequence[Table], Sequence[Sheet]]],
    kind: Callable[[str], str],
) -> list[Found]:
    """Every table with figures in `readings`, with its kind and whether an audit flags it."""
    out: list[Found] = []
    for name in readings:
        tables, read = load(name)
        for sheet in read:
            out.append(
                Found(name, sheet.index, kind(name), audit(tables[sheet.index], sheet).flagged())
            )
    return out
