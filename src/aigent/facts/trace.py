"""A cross-checked cell traced back through its reading: its table's rows as parsed (a filed
PDF's body re-rowed by its printed lines) beside how the cross-check read each one, for a person
judging where a figure came from."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from cltrplusf.chunk import Document

from aigent.facts.cfg_facts import CROSSCHECK_TRACE_WINDOW
from aigent.facts.crosscheck import Checked, RowRead, Rule, Sheet
from aigent.facts.headers import Column


@dataclass(frozen=True)
class TracedRow:
    """One table row: its cells as parsed, whether Docling marked it a header row
    (captions count) or a section's start, and the cross-check's reading; None for its header."""

    row: int
    cells: tuple[str, ...]
    docling_header: bool
    section: bool
    read: RowRead | None


@dataclass(frozen=True)
class Trace:
    """A fact's cell in its table: the table's basis, scale and column periods as read, how many
    rows were header, the rule it agreed by, and the header rows then the rows around it."""

    fact_id: int
    document: str
    sheet: int
    page: int
    basis: str | None
    scale: float | None
    columns: dict[int, str]
    header: int
    row: int
    column: int
    printed: str
    rule: Rule | None
    share: float | None
    rows: tuple[TracedRow, ...]


def period_text(column: Column) -> str:
    """A column's period as read: "3 months to 2024-06-30", "at 2024-03-31", "ending ..."."""
    if column.end is None:
        return "no date read"
    if column.months == 0:
        return f"at {column.end.isoformat()}"
    if column.months is None:
        return f"ending {column.end.isoformat()}"
    return f"{column.months} months to {column.end.isoformat()}"


def trace(
    checked: Checked,
    document: Document,
    read: Sequence[Sheet],
    window: int = CROSSCHECK_TRACE_WINDOW,
) -> Trace:
    """`checked`'s cell in `document`, whose sheets are `read`. ValueError for a fact resting on no
    cell, a table not read as a sheet, or a cell no longer printing what the run recorded."""
    sheet_at, row, column = checked.sheet, checked.row, checked.column
    if sheet_at is None or row is None or column is None:
        raise ValueError(f"fact {checked.fact_id} rests on no cell ({checked.outcome})")
    sheet = next((s for s in read if s.index == sheet_at), None)
    if sheet is None:
        raise ValueError(f"{document.doc_id}: table {sheet_at} has no figures to read")
    table = document.tables[sheet_at]
    now = table.rows[row][column].strip() if row < len(table.rows) else ""
    if now != checked.printed:
        raise ValueError(
            f"{document.doc_id}: table {sheet_at} row {row} column {column} prints {now!r}, "
            f"not {checked.printed!r}; re-run the cross-check"
        )
    by_row = {r.row: r for r in sheet.rows}
    shown = [
        *range(sheet.header),
        *range(max(sheet.header, row - window), min(len(table.rows), row + window + 1)),
    ]
    return Trace(
        checked.fact_id,
        checked.document,
        sheet_at,
        sheet.page,
        sheet.basis,
        sheet.scale,
        {j: period_text(c) for j, c in sorted(sheet.columns.items())},
        sheet.header,
        row,
        column,
        now,
        checked.rule,
        checked.share,
        tuple(
            TracedRow(r, table.rows[r], r < table.header_rows, r in table.sections, by_row.get(r))
            for r in shown
        ),
    )


def trace_json(found: Trace) -> dict[str, object]:
    """A trace as plain JSON-ready data."""
    return asdict(found)
