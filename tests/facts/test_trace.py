"""Tracing a cross-checked cell back through its table, and the CLI that traces a run's facts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import pytest
from cltrplusf.chunk import Document, Table

from aigent.facts import __main__ as cli
from aigent.facts.crosscheck import Checked, Fact, RowRead, check_fact, sheets
from aigent.facts.headers import Column
from aigent.facts.trace import period_text, trace, trace_json

from .conftest import CROSSCHECK_Q1, CROSSCHECK_RESULTS, crosscheck_document

RESULTS = crosscheck_document("STANDALONE FINANCIAL RESULTS (₹ in crore)", CROSSCHECK_RESULTS)
_OTHER_INCOME = 70019200000.0  # row 3 of CROSSCHECK_RESULTS, at crore


_HEADING = " ".join(CROSSCHECK_RESULTS[1])  # the header's last row heads the first block


def _fact() -> Fact:
    return Fact(
        7,
        "ICICIBANK",
        "standalone",
        "OtherIncome",
        ("Other income",),
        None,
        CROSSCHECK_Q1[0],
        CROSSCHECK_Q1[1],
        _OTHER_INCOME,
        "INR",
        -5,
        "xbrl-ai",
        "results/ICICIBANK-2024-06-30.pdf",
    )


def _checked(document: Document = RESULTS) -> Checked:
    return check_fact(_fact(), sheets(document))


@pytest.mark.parametrize(
    ("window", "shown"),
    [
        pytest.param(0, [0, 1, 3], id="no window shows the header and the row"),
        pytest.param(1, [0, 1, 2, 3, 4], id="a row either side"),
        pytest.param(8, [0, 1, 2, 3, 4, 5, 6], id="a window past the table's ends stops at them"),
    ],
)
def test_a_trace_shows_the_header_then_the_rows_around_its_cell(
    window: int, shown: list[int]
) -> None:
    found = trace(_checked(), RESULTS, sheets(RESULTS), window)
    assert [r.row for r in found.rows] == shown, found.rows


def test_a_trace_pairs_each_row_with_how_it_was_read() -> None:
    found = trace(_checked(), RESULTS, sheets(RESULTS), 0)
    header, _, row = found.rows
    assert (header.read, header.cells) == (None, CROSSCHECK_RESULTS[0])
    assert row.read == RowRead(3, "figures", "Other income", "Other income", _HEADING, 0)
    where = (found.sheet, found.page, found.row, found.column, found.printed, found.header)
    assert where == (0, 1, 3, 2, "7,001.92", 2), found
    assert (found.basis, found.scale, found.rule) == ("standalone", 1e7, "label"), found
    assert found.columns == {
        2: "3 months to 2024-06-30",
        3: "3 months to 2024-03-31",
        4: "3 months to 2023-06-30",
        5: "12 months to 2024-03-31",
    }, found.columns


def test_a_trace_marks_the_rows_docling_flagged() -> None:
    [table] = RESULTS.tables
    flagged = replace(RESULTS, tables=(Table(table.start, table.end, table.rows, 1, (3,)),))
    found = trace(_checked(flagged), flagged, sheets(flagged), 0)
    marks = [(r.row, r.docling_header, r.section) for r in found.rows]
    assert marks == [(0, True, False), (1, False, False), (3, False, True)], marks


@pytest.mark.parametrize(
    ("change", "error"),
    [
        pytest.param(
            lambda c: replace(c, outcome="unmatched", sheet=None), "rests on no cell", id="no cell"
        ),
        pytest.param(
            lambda c: replace(c, sheet=4),
            "table 4 has no figures",
            id="a table without figures",
        ),
        pytest.param(
            lambda c: replace(c, printed="7,001.93"), "prints '7,001.92'", id="a cell changed since"
        ),
        pytest.param(lambda c: replace(c, row=40), "prints ''", id="a row past the table's end"),
    ],
)
def test_a_cell_that_cannot_be_traced_is_refused(
    change: Callable[[Checked], Checked], error: str
) -> None:
    checked = change(_checked())
    with pytest.raises(ValueError, match=error):
        trace(checked, RESULTS, sheets(RESULTS))


@pytest.mark.parametrize(
    ("column", "text"),
    [
        (Column(date(2024, 6, 30), 3), "3 months to 2024-06-30"),
        (Column(date(2024, 3, 31), 0), "at 2024-03-31"),
        (Column(date(2024, 3, 31), None), "ending 2024-03-31"),
        (Column(None, None), "no date read"),
    ],
)
def test_period_text(column: Column, text: str) -> None:
    assert period_text(column) == text


def test_the_cli_traces_a_runs_facts_in_order_reading_each_document_once(tmp_path: Path) -> None:
    traced, other = _checked(), replace(_checked(), fact_id=8, printed="1")
    rows = tmp_path / "run.rows.jsonl"
    rows.write_text("".join(json.dumps(asdict(c)) + "\n" for c in (traced, other)))
    loads: list[str] = []

    def load(name: str) -> Document:
        loads.append(name)
        return RESULTS

    out = cli.trace_facts(rows, [9, 8, 7], load)
    assert out[0] == {"fact_id": 9, "error": "not in run.rows.jsonl"}, out[0]
    assert out[1]["fact_id"] == 8 and "prints '7,001.92'" in str(out[1]["error"]), out[1]
    assert out[2] == trace_json(trace(traced, RESULTS, sheets(RESULTS))), out[2]
    assert loads == [traced.document], loads
