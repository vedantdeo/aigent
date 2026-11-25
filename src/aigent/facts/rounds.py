"""The table-check round as code: two cross-check runs compared fact by fact (`diff_runs`), the
tables the most facts differ in (`rank_tables`), and every table's audit flags, one document a
process so memory is returned between them (`audit_tables`)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

Row = dict[str, object]


def read_rows(path: Path) -> dict[int, Row]:
    """A run's `.rows.jsonl` by fact id."""
    out: dict[int, Row] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row: Row = json.loads(line)
            out[int(str(row["fact_id"]))] = row
    return out


@dataclass(frozen=True)
class RunDiff:
    """Two runs fact by fact: agreements lost and won, how many of each were zero-valued facts,
    the outcomes before and after, and every other field that changed on a fact, counted."""

    lost: tuple[int, ...]
    won: tuple[int, ...]
    lost_zero: int
    won_zero: int
    outcomes: tuple[Counter[str], Counter[str]]
    fields: Counter[str]


def _agreed(row: Row) -> bool:
    return row.get("outcome") == "agree"


def diff_runs(before: Mapping[int, Row], after: Mapping[int, Row]) -> RunDiff:
    """`after` against `before` over the facts both hold; `veto`, a field older runs lack, is
    compared only where both have it."""
    shared = sorted(before.keys() & after.keys())
    lost = tuple(f for f in shared if _agreed(before[f]) and not _agreed(after[f]))
    won = tuple(f for f in shared if not _agreed(before[f]) and _agreed(after[f]))
    fields: Counter[str] = Counter()
    for f in shared:
        old, new = before[f], after[f]
        for key in old.keys() | new.keys():
            if key == "veto" and not (key in old and key in new):
                continue
            if old.get(key) != new.get(key):
                fields[key] += 1
    return RunDiff(
        lost,
        won,
        sum(before[f].get("value") == 0 for f in lost),
        sum(after[f].get("value") == 0 for f in won),
        (
            Counter(str(before[f].get("outcome")) for f in shared),
            Counter(str(after[f].get("outcome")) for f in shared),
        ),
        fields,
    )


def diff_report(diff: RunDiff, before: Mapping[int, Row], after: Mapping[int, Row]) -> str:
    """A diff as lines a person reads: the totals, the fields that moved, and the tables and
    vetoes the lost and won agreements gathered in, most first."""

    def where(rows: Mapping[int, Row], facts: Iterable[int]) -> Counter[str]:
        return Counter(f"{rows[f].get('document')} #{rows[f].get('sheet')}" for f in facts)

    lines = [
        f"agree {diff.outcomes[0]['agree']} -> {diff.outcomes[1]['agree']}; "
        f"differ {diff.outcomes[0]['differ']} -> {diff.outcomes[1]['differ']}; "
        f"unmatched {diff.outcomes[0]['unmatched']} -> {diff.outcomes[1]['unmatched']}",
        f"lost {len(diff.lost)} ({diff.lost_zero} zero-valued), "
        f"won {len(diff.won)} ({diff.won_zero} zero-valued)",
        "fields changed: "
        + (", ".join(f"{k} {n}" for k, n in diff.fields.most_common()) or "none"),
    ]
    for title, counted in (
        ("lost from", where(before, diff.lost)),
        ("won in", where(after, diff.won)),
        (
            "lost to vetoes",
            Counter(str(after[f].get("veto") or "").split(":")[0] for f in diff.lost),
        ),
    ):
        top = [f"  {n} {k}" for k, n in counted.most_common(10) if k]
        if top:
            lines += [title + ":", *top]
    return "\n".join(lines)


@dataclass(frozen=True)
class RankedTable:
    """A table by how many facts differ in it and agree in it."""

    document: str
    sheet: int
    differ: int
    agree: int


def rank_tables(rows: Iterable[Row], top: int, readings: bool = True) -> list[RankedTable]:
    """The `top` tables the most facts differ in, most first, ties by document and index;
    re-OCR'd readings left out unless `readings`."""
    differ: Counter[tuple[str, int]] = Counter()
    agree: Counter[tuple[str, int]] = Counter()
    for row in rows:
        document, sheet = str(row.get("document")), row.get("sheet")
        if sheet is None or (not readings and "#" in document):
            continue
        key = (document, int(str(sheet)))
        if row.get("outcome") == "differ":
            differ[key] += 1
        elif row.get("outcome") == "agree":
            agree[key] += 1
    ranked = sorted(differ.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [RankedTable(d, s, n, agree[(d, s)]) for (d, s), n in ranked]


def _audit_one(reading: str) -> tuple[str, dict[str, dict[str, object]]]:
    """One reading's tables as read, with their audit flags; run in a process of its own."""
    from aigent.facts.crosscheck import sheets
    from aigent.facts.documents import load_verify_document
    from aigent.facts.table_page import audit

    document = load_verify_document(reading)
    out: dict[str, dict[str, object]] = {}
    for sheet in sheets(document):
        table = document.tables[sheet.index]
        out[f"{reading}#{sheet.index}"] = {
            "rows": table.rows,
            "cells": len(sheet.cells),
            "columns": {str(j): str(c) for j, c in sheet.columns.items()},
            "flags": audit(table, sheet).counts(),
        }
    return reading, out


def audit_tables(readings: Sequence[str]) -> dict[str, dict[str, object]]:
    """Every table of `readings` as read, with its audit flags, by `<reading>#<index>`; one
    reading a fresh process, one at a time, so a reading's memory is returned before the next."""
    found: dict[str, dict[str, object]] = {}
    with get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        for _, tables in pool.imap(_audit_one, readings):
            found.update(tables)
    return found
