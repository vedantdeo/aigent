"""The XBRL cross-check: printed numbers read, column periods, scale and basis, each fact's outcome,
the tier it moves to, the sample for a person and the report."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import psutil
import pytest

from aigent.facts import __main__ as cli
from aigent.facts.crosscheck import (
    Column,
    Fact,
    agrees,
    apply,
    check_fact,
    column_periods,
    crosscheck,
    document_scale,
    documents_for,
    facts_to_check,
    label_share,
    number,
    sheets,
)
from aigent.facts.documents import Entry
from aigent.facts.store import FilingRow, sync_xbrl_facts, upsert_filing
from aigent.facts.xbrl import Fact as XbrlFact
from aigent.retrieval.chunk import Document, Table

from .conftest import CROSSCHECK_Q1, CROSSCHECK_WRITER, checked_row

Q1 = CROSSCHECK_Q1
NOW = datetime(2026, 11, 1, 9, 0)

_HEADER = (
    ("", "", "Quarter ended", "", "", "Year ended"),
    ("Sr", "Particulars", "30.06.2024", "31.03.2024", "30.06.2023", "31.03.2024"),
)
_RESULTS = (
    *_HEADER,
    ("1", "Interest earned", "38,995.78", "37,948.36", "33,327.61", "142,890.94"),
    ("2", "Other income", "7,001.92", "5,648.78", "5,435.25", "22,957.77"),
    ("3", "Exceptional items", "..", "..", "..", ".."),
    ("4", "Basic EPS", "15.73", "15.26", "13.80", "58.38"),
    ("5", "Gross NPA ratio", "2.15%", "2.16%", "2.76%", "2.16%"),
)


def _document(*parts: str | tuple[tuple[str, ...], ...]) -> Document:
    """A document of text and tables in order, each table printed one row a line."""
    text, tables = "", []
    for part in parts:
        if isinstance(part, str):
            text += part + "\n"
            continue
        start = len(text)
        text += "\n".join(" ".join(c for c in row if c) for row in part) + "\n"
        tables.append(Table(start, len(text), part))
    return Document("D", text, page_starts=(0,), tables=tuple(tables))


def _fact(
    concept: str = "InterestEarned",
    value: float = 389957800000.0,
    labels: tuple[str, ...] = ("Interest earned",),
    basis: str = "standalone",
    period: tuple[date | None, date] = Q1,
    unit: str = "INR",
    decimals: int | None = -5,
    segment: str | None = None,
) -> Fact:
    return Fact(
        1,
        "ICICIBANK",
        basis,
        concept,
        labels,
        segment,
        period[0],
        period[1],
        value,
        unit,
        decimals,
        "xbrl-ai",
        "results/ICICIBANK-2024-06-30.pdf",
    )


STANDALONE = _document("STANDALONE FINANCIAL RESULTS (₹ in crore)", _RESULTS)


@pytest.mark.parametrize(
    ("printed", "read"),
    [
        ("38,995.78", (38995.78, 2, False)),
        ("1,07,000", (107000.0, 0, False)),
        ("59923,95", (59923.95, 2, False)),
        ("1.407.45", (1407.45, 2, False)),
        ("2652. 72", (2652.72, 2, False)),
        (".. 14,692.66", (14692.66, 2, False)),
        ("(1,937.10)", (-1937.10, 2, False)),
        ("(29642.01", (-29642.01, 2, False)),
        ("-12.5", (-12.5, 1, False)),
        ("2.15%", (2.15, 2, True)),
        ("..", (0.0, 0, False)),
        ("-", (0.0, 0, False)),
        ("—", (0.0, 0, False)),
        pytest.param("15.73 15.46", None, id="two numbers in a cell"),
        pytest.param("31.03.2025", None, id="a date"),
        pytest.param("June 30, 2024", None, id="a worded date"),
        pytest.param("Interest earned", None, id="text"),
        pytest.param("", None, id="an empty cell"),
    ],
)
def test_number(printed: str, read: tuple[float, int, bool] | None) -> None:
    assert number(printed) == read


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        pytest.param(
            _HEADER,
            {
                2: Column(date(2024, 6, 30), 3),
                3: Column(date(2024, 3, 31), 3),
                4: Column(date(2023, 6, 30), 3),
                5: Column(date(2024, 3, 31), 12),
            },
            id="a group heading over dated columns",
        ),
        pytest.param(
            (
                ("Six months ended",),
                ("", "Three months ended", "", "Six months ended", "Year ended"),
                (
                    "Particulars",
                    "September 30, 2025 (Q2-2026)",
                    "June 30, 2025 (Q1-2026)",
                    "September 30, 2025 (H1-2026)",
                    "March 31, 2025 (FY2025)",
                ),
            ),
            {
                1: Column(date(2025, 9, 30), 3),
                2: Column(date(2025, 6, 30), 3),
                3: Column(date(2025, 9, 30), 6),
                4: Column(date(2025, 3, 31), 12),
            },
            id="a column's own cells over a stray heading",
        ),
        pytest.param(
            (("Particulars", "As at 31.03.2025", "Asat 31.03.2024"),),
            {1: Column(date(2025, 3, 31), 0), 2: Column(date(2024, 3, 31), 0)},
            id="balances as at a day",
        ),
        pytest.param(
            (("", "Nine months ended", ""), ("Particulars", "31.12.2024", "31.12.2023")),
            {1: Column(date(2024, 12, 31), 9), 2: Column(date(2023, 12, 31), 9)},
            id="nine months",
        ),
        pytest.param(
            (("Particulars", "31.12.2024"),),
            {1: Column(date(2024, 12, 31), None)},
            id="a date with no length",
        ),
        pytest.param(
            (
                ("Particulars", "Quarter Ended", "", "", "Half Year Ended"),
                ("", "30 Sep'24", "30 Jun'24", "30 Sep'23", "30 Sep'24"),
            ),
            {
                1: Column(date(2024, 9, 30), 3),
                2: Column(date(2024, 6, 30), 3),
                3: Column(date(2023, 9, 30), 3),
                4: Column(date(2024, 9, 30), 6),
            },
            id="Reliance's day, month and two-digit year",
        ),
        pytest.param(
            (("", "Quarter Ended", "", ""), ("", "3osea'24", "30 Sen'23", "30 Seo'24")),
            {
                1: Column(date(2024, 9, 30), 3),
                2: Column(date(2023, 9, 30), 3),
                3: Column(date(2024, 9, 30), 3),
            },
            id="an OCR'd month read by the two letters only September starts with",
        ),
        pytest.param(
            (("", "Quarter Ended", ""), ("", "30 Jnu'24", "31 Mav'24")),
            {},
            id="an OCR'd month two months start like is left undated",
        ),
        pytest.param(
            (
                ("Particulars", "Quarter Ended", "", "", "Half Year Ended", "", "Year Ended"),
                (
                    "",
                    "30thSeo'25",
                    "30 th Jun'25",
                    "30 th Sep'24",
                    "30th Sep'25",
                    "30th Seo'24",
                    "31 st Mar'25",
                ),
            ),
            {
                1: Column(date(2025, 9, 30), 3),
                2: Column(date(2025, 6, 30), 3),
                3: Column(date(2024, 9, 30), 3),
                4: Column(date(2025, 9, 30), 6),
                5: Column(date(2024, 9, 30), 6),
                6: Column(date(2025, 3, 31), 12),
            },
            id="Reliance's ordinal days",
        ),
        pytest.param(
            (
                ("", "Quarter Ended", "", "Half Year Ended"),
                ("Particulars", "30.09.2025", "30.06.2025", "30.09.2025"),
                (
                    "Earnings per equity share (Not Annualised for the quarter/half year)",
                    "",
                    "",
                    "",
                ),
            ),
            {
                1: Column(date(2025, 9, 30), 3),
                2: Column(date(2025, 6, 30), 3),
                3: Column(date(2025, 9, 30), 6),
            },
            id="a title in the label column carries over no column",
        ),
        pytest.param(
            (("", "", "(₹ in crores)", ""), ("", "PROFITS", "2024 - 25", "2023 - 24")),
            {2: Column(date(2025, 3, 31), 12), 3: Column(date(2024, 3, 31), 12)},
            id="ITC's financial years",
        ),
        pytest.param(
            (("Particulars", "FY25", "FY 2024"),),
            {1: Column(date(2025, 3, 31), 12), 2: Column(date(2024, 3, 31), 12)},
            id="FY short forms",
        ),
        pytest.param(
            (("Particulars", "2019-24"),),
            {},
            id="a span of years is no financial year",
        ),
        pytest.param(
            (
                ("", "", "For the year ended", ""),
                ("Particulars", "Note No.", "2025", "(₹ In Crore) 31 March 2024"),
            ),
            {2: Column(date(2025, 3, 31), 12), 3: Column(date(2024, 3, 31), 12)},
            id="Bajaj Auto's bare year takes its neighbour's day",
        ),
        pytest.param(
            (("Particulars", "2025", "2024"),),
            {},
            id="bare years with no dated neighbour stay undated",
        ),
        pytest.param(
            (("", "Year ended", "", ""), ("Particulars", "2025", "31 March 2024", "31.12.2023")),
            {2: Column(date(2024, 3, 31), 12), 3: Column(date(2023, 12, 31), 12)},
            id="a bare year among neighbours of different days stays undated",
        ),
        pytest.param(
            (("Particulars", "Year ended 31st March, 2025", "As at 31 st March 2024"),),
            {1: Column(date(2025, 3, 31), 12), 2: Column(date(2024, 3, 31), 0)},
            id="the day first, as annual reports print it",
        ),
        pytest.param(
            (("Particulars", "December 2023", "31.03.2024"),),
            {2: Column(date(2024, 3, 31), None)},
            id="a month and year with no day takes no neighbour's day",
        ),
        pytest.param(
            (
                ("(₹ in crores)",),
                ("2024",),
                ("", "", "As at March 31, 2025", "As at March 31,"),
                ("", "", "2024", ""),
            ),
            {2: Column(date(2025, 3, 31), 0)},
            id="Tata Motors' year on a line of its own is a wrapped title, not a column",
        ),
        pytest.param(
            (("Particulars", "31.03.2025", "31.03.2024"), ("Restated", "2024", "")),
            {1: Column(date(2025, 3, 31), None), 2: Column(date(2024, 3, 31), None)},
            id="a dated column keeps its date over a stray year",
        ),
    ],
)
def test_column_periods(header: tuple[tuple[str, ...], ...], expected: dict[int, Column]) -> None:
    assert column_periods(header) == expected


@pytest.mark.parametrize(
    ("text", "scale"),
    [
        ("(₹ in crore) ... (₹ in crore) ... in lac", 1e7),
        ("(₹ in lac) figures in lakhs", 1e5),
        ("USD in million", 1e6),
        pytest.param("the place where it was replaced", None, id="lac inside a word"),
        ("no scale here", None),
    ],
)
def test_document_scale(text: str, scale: float | None) -> None:
    assert document_scale(text) == scale


def test_a_sheet_takes_its_basis_and_scale_from_the_text_before_it() -> None:
    document = _document(
        "STANDALONE FINANCIAL RESULTS (₹ in crore)",
        _RESULTS,
        "Segment results",
        _RESULTS,
        "CONSOLIDATED FINANCIAL RESULTS (₹ in lac)",
        _RESULTS,
    )
    read = sheets(document)
    assert [(s.basis, s.scale) for s in read] == [
        ("standalone", 1e7),
        ("standalone", 1e7),
        ("consolidated", 1e5),
    ]


def test_a_sheet_with_no_scale_near_it_takes_the_documents() -> None:
    document = _document("STANDALONE RESULTS", _RESULTS, "Notes (₹ in crore) and (₹ in crore)")
    assert sheets(document)[0].scale == 1e7


def _pages(*tables: tuple[int, tuple[tuple[str, ...], ...]]) -> Document:
    """A document of tables, each on the page given, its rows printed one a line."""
    text, starts, parts = "", [0], []
    for page, rows in tables:
        while len(starts) < page:
            text += "\f"
            starts.append(len(text))
        start = len(text)
        text += "\n".join(" ".join(c for c in row if c) for row in rows) + "\n"
        parts.append(Table(start, len(text), rows))
    return Document("D", text, page_starts=tuple(starts), tables=tuple(parts))


_STATEMENT = (
    ("", "Quarter ended", "", "Year ended"),
    ("Particulars", "30.09.2024", "30.06.2024", "31.03.2024"),
    ("Revenue", "100", "90", "380"),
)
_CONTINUED = (("Basic EPS", "2.50", "3.53", "10.44"), ("Diluted EPS", "2.49", "3.52", "10.40"))


@pytest.mark.parametrize(
    ("later", "inherits"),
    [
        pytest.param((1, _CONTINUED), True, id="the rest of a statement on the same page"),
        pytest.param((2, _CONTINUED), True, id="the rest of a statement on the next page"),
        pytest.param((3, _CONTINUED), False, id="a table two pages on"),
        pytest.param(
            (2, (("Basic EPS", "2.50", "3.53", "10.44", "9.80"),)),
            False,
            id="a table of another width",
        ),
        pytest.param(
            (2, (("Particulars", "31.12.2024", "30.09.2024", "31.03.2024"), *_CONTINUED)),
            False,
            id="a table with dated columns of its own",
        ),
    ],
)
def test_a_table_continuing_a_statement_takes_its_columns(
    later: tuple[int, tuple[tuple[str, ...], ...]], inherits: bool
) -> None:
    first, second = sheets(_pages((1, _STATEMENT), later))
    assert (second.columns == first.columns) is inherits, second.columns


_TOTAL_EPS = "Basic earnings (loss) per share from continuing and discontinued operations"
_DILUTED_EPS = "Diluted earnings (loss) per share from continuing and discontinued operations"
_CONTINUING_EPS = "Basic earnings (loss) per share from continuing operations"
_DISCONTINUED_EPS = "Basic earnings (loss) per share from discontinued operations"
_WEIGHTED = "Weighted average number of equity shares for basic earnings per share"
_EPS_HEADING = ("", "Earnings per equity share", "", "", "", "")


def _eps_row(label: str) -> tuple[str, ...]:
    return ("", label, "15.73", "15.26", "13.80", "58.38")


@pytest.mark.parametrize(
    ("rows", "label", "unit", "agreed"),
    [
        pytest.param(
            (_EPS_HEADING, _eps_row("Basic (in ₹)")),
            _TOTAL_EPS,
            "INRPerShare",
            True,
            id="a short row under its heading",
        ),
        pytest.param(
            (_EPS_HEADING, _eps_row("~asic (in~)")),
            _TOTAL_EPS,
            "INRPerShare",
            True,
            id="an OCR'd kind one letter off",
        ),
        pytest.param(
            (("", "XIII Earnin11s oer share /EPS)", "", "", "", ""), _eps_row("Basic (in ₹)")),
            _TOTAL_EPS,
            "INRPerShare",
            True,
            id="an OCR'd heading still naming EPS",
        ),
        pytest.param(
            (_EPS_HEADING, _eps_row("Basic (in ₹)")),
            _DILUTED_EPS,
            "INRPerShare",
            False,
            id="a diluted fact on a basic row",
        ),
        pytest.param(
            (_EPS_HEADING, _eps_row("Basic and diluted (in ₹)")),
            _DILUTED_EPS,
            "INRPerShare",
            True,
            id="a row printing both kinds",
        ),
        pytest.param(
            (("", "Other income", "", "", "", ""), _eps_row("Basic (in ₹)")),
            _TOTAL_EPS,
            "INRPerShare",
            False,
            id="no earnings per share above it",
        ),
        pytest.param(
            (_EPS_HEADING, _eps_row("Basic EPS from discontinued operations")),
            _CONTINUING_EPS,
            "INRPerShare",
            False,
            id="a continuing fact on a discontinued row",
        ),
        pytest.param(
            (_EPS_HEADING, _eps_row("Basic (in ₹)")),
            _DISCONTINUED_EPS,
            "INRPerShare",
            False,
            id="a discontinued fact on a plain row",
        ),
        pytest.param(
            (
                ("", "Earnings per share from discontinued operations", "", "", "", ""),
                _eps_row("Basic"),
            ),
            _DISCONTINUED_EPS,
            "INRPerShare",
            True,
            id="a discontinued fact under a discontinued heading",
        ),
        pytest.param(
            (_EPS_HEADING, _eps_row("Basic EPS from continuing and discontinued operations")),
            _TOTAL_EPS,
            "INRPerShare",
            True,
            id="a total fact on a row naming both",
        ),
        pytest.param(
            (_EPS_HEADING, _eps_row("Baslo (in ₹)")),
            _TOTAL_EPS,
            "INRPerShare",
            False,
            id="a word two letters off is no kind",
        ),
        pytest.param(
            (_EPS_HEADING, _eps_row("Basic (in ₹)")),
            _WEIGHTED,
            "shares",
            False,
            id="a fact that is no per-share figure",
        ),
    ],
)
def test_a_per_share_fact_agrees_on_its_kind_of_row(
    rows: tuple[tuple[str, ...], ...], label: str, unit: str, agreed: bool
) -> None:
    scaled = ("", "STANDALONE RESULTS (₹ in crore)", "", "", "", "")
    figures = ("1", "Interest earned", "38,995.78", "37,948.36", "33,327.61", "1")
    read = sheets(_pages((1, (scaled, *_HEADER, figures, *rows))))
    fact = _fact("EPS", 15.73 if unit == "INRPerShare" else 157_300_000.0, (label,), unit=unit)
    assert (check_fact(fact, read).outcome == "agree") is agreed, check_fact(fact, read)


def test_a_heading_printed_above_the_first_figures_counts() -> None:
    header = (*_HEADER, _EPS_HEADING)
    read = sheets(_pages((1, (*header, _eps_row("a) Basic (in ₹)")))))
    fact = _fact("EPS", 15.73, (_TOTAL_EPS,), unit="INRPerShare", decimals=2)
    assert check_fact(fact, read).outcome == "agree", read[0].cells[:1]


def test_a_heading_carries_into_the_statement_it_continues() -> None:
    first = (*_HEADER, ("1", "Interest earned", "1.00", "2.00", "3.00", "4.00"), _EPS_HEADING)
    read = sheets(_pages((1, first), (2, (_eps_row("Basic (in ₹)"),))))
    fact = _fact("EPS", 15.73, (_TOTAL_EPS,), unit="INRPerShare", decimals=2)
    assert check_fact(fact, read).outcome == "agree", read[-1].cells[:1]


def test_a_rows_serial_number_is_not_a_figure() -> None:
    [sheet] = sheets(STANDALONE)
    assert all(c.column != 0 for c in sheet.cells)
    assert sheet.cells[0].label == "Interest earned"


def test_a_row_sitting_a_column_off_its_header_is_read_by_order() -> None:
    shifted = (*_HEADER, ("Interest earned", "38,995.78", "37,948.36", "33,327.61", "142,890.94"))
    [sheet] = sheets(_document("STANDALONE (₹ in crore)", shifted))
    first = sheet.cells[0]
    assert (first.column, first.period) == (1, Column(date(2024, 6, 30), 3))


def test_a_row_short_of_figures_is_read_by_its_own_columns() -> None:
    gappy = (*_HEADER, ("1", "Interest earned", "38,995.78", "", "", "142,890.94"))
    [sheet] = sheets(_document("STANDALONE (₹ in crore)", gappy))
    assert [c.period for c in sheet.cells] == [
        Column(date(2024, 6, 30), 3),
        Column(date(2024, 3, 31), 12),
    ]


@pytest.mark.parametrize(
    ("fact", "printed", "scale", "agreed"),
    [
        pytest.param(_fact(), "38,995.78", 1e7, True, id="rupees at the table's scale"),
        pytest.param(_fact(value=389958000000.0), "38,995.78", 1e7, False, id="a lakh out"),
        pytest.param(_fact(), "38,995.78", 1e5, False, id="the wrong scale"),
        pytest.param(_fact(value=38995.78), "38,995.78", None, False, id="no scale known"),
        pytest.param(_fact(unit="INRPerShare", value=15.73), "15.73", 1e7, True, id="per share"),
        pytest.param(_fact(unit="pure", value=0.0215), "2.15%", 1e7, True, id="a percentage"),
        pytest.param(_fact(unit="pure", value=0.0002), "2.15%", 1e7, False, id="filed /100"),
        pytest.param(_fact(unit="pure", value=0.41), "0.41", 1e7, True, id="a plain ratio"),
        pytest.param(_fact(value=21500000.0), "2.15%", 1e7, False, id="a percentage for rupees"),
    ],
)
def test_agrees(fact: Fact, printed: str, scale: float | None, agreed: bool) -> None:
    [sheet] = sheets(_document("STANDALONE", (*_HEADER, ("1", "x", printed, "", "", ""))))
    assert agrees(fact, sheet.cells[0], scale) is agreed


@pytest.mark.parametrize(
    ("fact", "outcome", "printed"),
    [
        pytest.param(_fact(), "agree", "38,995.78", id="its labelled row"),
        pytest.param(
            _fact("OtherIncome", 70019200000.0, ("Income not from interest",)),
            "agree",
            "7,001.92",
            id="a distinctive figure in an unlabelled row",
        ),
        pytest.param(
            _fact("ExceptionalItems", 0.0, ("Exceptional items",)),
            "agree",
            "..",
            id="a zero printed as dots",
        ),
        pytest.param(
            _fact("Eps", 15.73, ("Diluted EPS",), unit="INRPerShare"),
            "unmatched",
            None,
            id="a short figure needs its label",
        ),
        pytest.param(
            _fact(value=390000000000.0), "differ", "38,995.78", id="its row prints another figure"
        ),
        pytest.param(
            _fact(period=(date(2023, 4, 1), date(2024, 3, 31)), value=1428909400000.0),
            "agree",
            "142,890.94",
            id="the year's column",
        ),
        pytest.param(
            _fact(period=(date(2024, 1, 1), date(2024, 3, 31)), value=1428909400000.0),
            "differ",
            "37,948.36",
            id="the year's figure filed for the quarter",
        ),
        pytest.param(
            _fact(period=(None, date(2024, 3, 31)), value=379483600000.0),
            "agree",
            "37,948.36",
            id="a balance in any column ending that day",
        ),
        pytest.param(_fact(basis="consolidated"), "unmatched", None, id="another basis's table"),
        pytest.param(
            _fact("Npa", 0.0215, ("Gross NPA ratio",), unit="pure"),
            "agree",
            "2.15%",
            id="a ratio as a percentage",
        ),
        pytest.param(
            _fact("Unprinted", 123456789.0, ("Something never printed",)),
            "unmatched",
            None,
            id="nothing like it printed",
        ),
    ],
)
def test_check_fact(fact: Fact, outcome: str, printed: str | None) -> None:
    checked = check_fact(fact, sheets(STANDALONE))
    assert (checked.outcome, checked.printed) == (outcome, printed), checked


def test_a_fact_records_where_it_agreed() -> None:
    checked = check_fact(_fact(), sheets(STANDALONE))
    assert (checked.page, checked.row_label) == (1, "Interest earned")


FLAGGED: Entry = {"format": "pdf", "pages": 2, "text_less_pages": [], "ocr_pages": []}
MANIFEST: dict[str, Entry] = {
    "results/ICICIBANK-2024-06-30.pdf": FLAGGED,
    "results/HDFCBANK-2024-06-30.pdf": FLAGGED,
}


def test_facts_to_check_are_the_companies_named_with_a_document(
    crosscheck_db: sqlite3.Connection,
) -> None:
    facts = facts_to_check(crosscheck_db, ("ICICIBANK",), MANIFEST)
    assert [(f.basis, f.concept, f.document) for f in facts] == [
        ("consolidated", "InterestEarned", "results/ICICIBANK-2024-06-30.pdf"),
        ("standalone", "InterestEarned", "results/ICICIBANK-2024-06-30.pdf"),
        ("standalone", "OtherIncome", "results/ICICIBANK-2024-06-30.pdf"),
    ]
    assert facts[0].labels[1:] == ("Interest income",)


def test_facts_to_check_skip_a_fact_not_from_xbrl(crosscheck_db: sqlite3.Connection) -> None:
    crosscheck_db.execute("UPDATE facts SET tier = 'filed-cell-ai' WHERE concept = 'OtherIncome'")
    facts = facts_to_check(crosscheck_db, ("ICICIBANK",), MANIFEST)
    assert "OtherIncome" not in {f.concept for f in facts}


@pytest.mark.parametrize(
    ("period", "filed", "stems"),
    [
        pytest.param(Q1, date(2024, 6, 30), ("ICICIBANK-2024-06-30",), id="a quarter"),
        pytest.param(
            (date(2023, 4, 1), date(2024, 3, 31)),
            date(2024, 3, 31),
            ("ICICIBANK-FY24", "ICICIBANK-2024-03-31"),
            id="a year's figure tries its annual report first",
        ),
        pytest.param(
            (None, date(2025, 3, 31)),
            date(2025, 3, 31),
            ("ICICIBANK-FY25", "ICICIBANK-2025-03-31"),
            id="a 31 March balance",
        ),
        pytest.param(
            (date(2025, 1, 1), date(2025, 3, 31)),
            date(2025, 3, 31),
            ("ICICIBANK-2025-03-31",),
            id="the fourth quarter alone is not in the annual report",
        ),
        pytest.param(
            (None, date(2024, 3, 31)),
            date(2025, 3, 31),
            ("ICICIBANK-2025-03-31",),
            id="a balance at another day than the filing's",
        ),
        pytest.param(
            (None, date(2024, 9, 30)),
            date(2024, 9, 30),
            ("ICICIBANK-2024-09-30",),
            id="a half-year balance",
        ),
        pytest.param(
            (date(2024, 4, 1), date(2024, 12, 31)),
            date(2024, 12, 31),
            ("ICICIBANK-2024-12-31",),
            id="nine months",
        ),
    ],
)
def test_documents_for(
    period: tuple[date | None, date], filed: date, stems: tuple[str, ...]
) -> None:
    assert documents_for(_fact(period=period), filed) == stems


_OCR_SCAN: Entry = {
    "format": "pdf",
    "pages": 10,
    "text_less_pages": [],
    "ocr_pages": list(range(2, 11)),
}
_IMAGE_SCAN: Entry = {
    "format": "pdf",
    "pages": 10,
    "text_less_pages": list(range(2, 11)),
    "ocr_pages": [],
}


@pytest.mark.parametrize(
    ("entry", "ocr_tickers", "checked"),
    [
        pytest.param(FLAGGED, (), True, id="a text pdf"),
        pytest.param({}, (), False, id="an entry not flagged"),
        pytest.param(_OCR_SCAN, ("ICICIBANK",), True, id="an ocr'd scan of a company read so"),
        pytest.param(_OCR_SCAN, ("HDFCBANK",), False, id="an ocr'd scan of another company"),
        pytest.param(_IMAGE_SCAN, ("ICICIBANK",), False, id="a scan with no text"),
    ],
)
def test_facts_to_check_leave_out_a_document_that_cannot_be_read(
    crosscheck_db: sqlite3.Connection, entry: Entry, ocr_tickers: tuple[str, ...], checked: bool
) -> None:
    manifest = {"results/ICICIBANK-2024-06-30.pdf": entry}
    facts = facts_to_check(crosscheck_db, ("ICICIBANK",), manifest, ocr_tickers)
    assert bool(facts) is checked, facts


_YEAR = (date(2023, 4, 1), date(2024, 3, 31))
_ANNUAL, _MARCH = "annual/ICICIBANK-FY24.pdf", "results/ICICIBANK-2024-03-31.pdf"


def _year_column(printed: str) -> Document:
    rows = (*_HEADER, ("1", "Interest earned", "1.00", "1.00", "1.00", printed))
    return _document("STANDALONE FINANCIAL RESULTS (₹ in crore)", rows)


_AGREES, _DIFFERS = _year_column("142,890.94"), _year_column("142,000.00")
_SILENT = _document("STANDALONE FINANCIAL RESULTS (₹ in crore)")


@pytest.mark.parametrize(
    ("annual", "results", "outcome", "document", "loaded"),
    [
        pytest.param(_AGREES, _DIFFERS, "agree", _ANNUAL, [_ANNUAL], id="the annual report agrees"),
        pytest.param(
            _DIFFERS, _AGREES, "agree", _MARCH, [_ANNUAL, _MARCH], id="the results agree after"
        ),
        pytest.param(
            _SILENT, _DIFFERS, "differ", _MARCH, [_ANNUAL, _MARCH], id="a difference beats silence"
        ),
        pytest.param(
            _DIFFERS, _DIFFERS, "differ", _ANNUAL, [_ANNUAL, _MARCH], id="a tie keeps the first"
        ),
    ],
)
def test_crosscheck_keeps_each_facts_best_outcome_across_its_documents(
    crosscheck_db: sqlite3.Connection,
    tmp_path: Path,
    annual: Document,
    results: Document,
    outcome: str,
    document: str,
    loaded: list[str],
) -> None:
    fid = upsert_filing(
        crosscheck_db,
        FilingRow("ICICIBANK", 2024, date(2024, 3, 31), "standalone", "xbrl", "march", "s", "t"),
    )
    year = XbrlFact("InterestEarned", "duration", *_YEAR, None, 1428909400000.0, "INR", -5)
    sync_xbrl_facts(crosscheck_db, fid, "ICICIBANK", "standalone", [year], CROSSCHECK_WRITER)
    manifest = {_ANNUAL: FLAGGED, _MARCH: FLAGGED}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    read: list[str] = []

    def load(name: str, verify_dir: Path) -> Document:
        read.append(name)
        return annual if name == _ANNUAL else results

    checked = crosscheck(crosscheck_db, ("ICICIBANK",), tmp_path, load)
    assert [(c.outcome, c.document) for c in checked] == [(outcome, document)], checked
    assert read == loaded


def test_crosscheck_reads_each_document_once(
    crosscheck_db: sqlite3.Connection, tmp_path: Path
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    loaded: list[str] = []

    def load(name: str, verify_dir: Path) -> Document:
        loaded.append(name)
        return STANDALONE

    checked = crosscheck(crosscheck_db, ("ICICIBANK",), tmp_path, load)
    assert loaded == ["results/ICICIBANK-2024-06-30.pdf"]
    assert [(c.basis, c.concept, c.outcome) for c in checked] == [
        ("consolidated", "InterestEarned", "unmatched"),
        ("standalone", "InterestEarned", "agree"),
        ("standalone", "OtherIncome", "differ"),
    ]


ORIGINS = {"results/ICICIBANK-2024-06-30.pdf": "nse-pdf"}


def test_apply_makes_only_agreements_xbrl_filed(crosscheck_db: sqlite3.Connection) -> None:
    ids = [r[0] for r in crosscheck_db.execute("SELECT id FROM facts ORDER BY id")]
    checked = [checked_row(ids[0], "agree"), checked_row(ids[1], "differ")]
    moved = apply(crosscheck_db, checked, NOW, ORIGINS)
    rows = crosscheck_db.execute(
        "SELECT tier, verified_by, verified_at, printed, page FROM facts ORDER BY id"
    ).fetchall()
    assert moved == 1
    assert rows[0] == (
        "xbrl-filed",
        "nse-pdf:results/ICICIBANK-2024-06-30.pdf",
        "2026-11-01T09:00:00",
        "38,995.78",
        3,
    )
    assert rows[1] == ("xbrl-ai", None, None, None, None)


def test_apply_leaves_a_persons_check_and_counts_nothing_twice(
    crosscheck_db: sqlite3.Connection,
) -> None:
    ids = [r[0] for r in crosscheck_db.execute("SELECT id FROM facts ORDER BY id")]
    crosscheck_db.execute(
        "UPDATE facts SET tier = 'xbrl', verified_by = 'vedant', verified_at = 't' WHERE id = ?",
        (ids[1],),
    )
    first = apply(
        crosscheck_db,
        [checked_row(ids[0], "agree"), checked_row(ids[1], "agree", "xbrl-ai")],
        NOW,
        ORIGINS,
    )
    again = apply(crosscheck_db, [checked_row(ids[0], "agree")], NOW, ORIGINS)
    assert (first, again) == (1, 0)
    assert crosscheck_db.execute(
        "SELECT tier, verified_by FROM facts WHERE id = ?", (ids[1],)
    ).fetchone() == (
        "xbrl",
        "vedant",
    )


@pytest.mark.parametrize(
    ("document", "origins", "stamp"),
    [
        pytest.param(
            "results/X.pdf", {"results/X.pdf": "nse-pdf"}, "nse-pdf:results/X.pdf", id="an nse pdf"
        ),
        pytest.param(
            "results/X.htm",
            {"results/X.htm": "sec-6k-html"},
            "sec-6k-html:results/X.htm",
            id="an sec copy",
        ),
    ],
)
def test_apply_stamps_each_agreement_with_its_documents_origin(
    crosscheck_db: sqlite3.Connection, document: str, origins: dict[str, str], stamp: str
) -> None:
    first = crosscheck_db.execute("SELECT min(id) FROM facts").fetchone()[0]
    apply(crosscheck_db, [replace(checked_row(first, "agree"), document=document)], NOW, origins)
    by = crosscheck_db.execute("SELECT verified_by FROM facts WHERE id = ?", (first,)).fetchone()
    assert by == (stamp,)


def test_apply_writes_nothing_for_a_document_with_no_origin(
    crosscheck_db: sqlite3.Connection,
) -> None:
    ids = [r[0] for r in crosscheck_db.execute("SELECT id FROM facts ORDER BY id")]
    unknown = replace(checked_row(ids[1], "agree"), document="results/NOPE.htm")
    with pytest.raises(KeyError, match="NOPE"):
        apply(crosscheck_db, [checked_row(ids[0], "agree"), unknown], NOW, ORIGINS)
    tiers = [r[0] for r in crosscheck_db.execute("SELECT tier FROM facts ORDER BY id")]
    assert set(tiers) == {"xbrl-ai"}, tiers


@pytest.mark.parametrize(("write", "tier"), [(False, "xbrl-ai"), (True, "xbrl-filed")])
def test_the_command_records_agreements_only_when_told(
    crosscheck_db: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write: bool,
    tier: str,
) -> None:
    db = Path(crosscheck_db.execute("PRAGMA database_list").fetchone()[2])
    first = crosscheck_db.execute("SELECT min(id) FROM facts").fetchone()[0]
    crosscheck_db.close()
    monkeypatch.setattr(cli, "FACTS_DB", db)
    monkeypatch.setattr(cli, "CROSSCHECK_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli, "crosscheck", lambda c, t: [checked_row(first, "agree")])
    manifest = {"results/ICICIBANK-2024-06-30.pdf": {"source": "nse", "format": "pdf"}}
    monkeypatch.setattr(cli, "read_manifest", lambda: manifest)
    cli.cross_check(NOW, write=write, decisions=tmp_path / "none.jsonl")
    after = (
        sqlite3.connect(db)
        .execute("SELECT tier, verified_by FROM facts WHERE id = ?", (first,))
        .fetchone()
    )
    assert after == (tier, "nse-pdf:results/ICICIBANK-2024-06-30.pdf" if write else None), after
    assert list((tmp_path / "reports").glob("crosscheck-*.md"))


@pytest.mark.parametrize(
    ("label", "row", "share"),
    [
        pytest.param("Interest earned", "1. Interest earned (a)+(b)", 1.0, id="the same words"),
        pytest.param(
            "Earnings per share", "Basic EPS", 1.0, id="a label naming neither takes either"
        ),
        pytest.param("Basic earnings per share", "a) Basic EPS", 1.0, id="EPS read in full"),
        pytest.param("Diluted EPS", "a) Basic EPS", 0.0, id="diluted never takes basic"),
        pytest.param("Basic EPS", "b) Diluted EPS", 0.0, id="basic never takes diluted"),
        pytest.param("Basic EPS", "Basic and diluted EPS", 1.0, id="a row naming both"),
        pytest.param("Gross NPA", "Net NPAs", 0.0, id="gross never takes net"),
        pytest.param("Gross NPA", "(a) Gross NPAs", 1.0, id="NPA read in full"),
        pytest.param("Interest earned", "Other income", 0.0, id="nothing shared"),
        pytest.param("Interest earned", "Interest expended", 0.5, id="half shared"),
        pytest.param(
            "Current tax",
            "Current tax on profits for the year discontinued operations",
            0.0,
            id="a row of discontinued operations alone takes no other fact",
        ),
        pytest.param(
            "Profit from discontinued operations",
            "Profit for the period",
            0.0,
            id="a fact of discontinued operations alone takes no other row",
        ),
        pytest.param(
            "Profit before tax",
            "Profit before tax from continuing operations",
            1.0,
            id="a total takes a continuing-operations row",
        ),
        pytest.param(
            "Profit from continuing and discontinued operations",
            "Profit from continuing and discontinued operations",
            1.0,
            id="both kinds on both sides",
        ),
    ],
)
def test_label_share(label: str, row: str, share: float) -> None:
    assert label_share([label], None, row) == pytest.approx(share), (label, row)


def test_a_segment_counts_as_part_of_the_label() -> None:
    assert label_share(["Segment revenue"], "Retail", "a Retail Banking") == pytest.approx(1 / 3)
    assert label_share(["Revenue"], "Retail", "a Retail Banking") == pytest.approx(1 / 2)


def test_the_best_of_a_facts_labels_counts() -> None:
    assert (
        label_share(
            ["Revenue on investments", "Income on investments"], None, "b) Income on investments"
        )
        == 1.0
    )


_OCR_ENTRY: Entry = {
    "source": "nse",
    "format": "pdf",
    "pages": 10,
    "text_less_pages": [],
    "ocr_pages": list(range(2, 11)),
}
_IMAGE_ENTRY: Entry = {**_OCR_ENTRY, "text_less_pages": list(range(2, 11)), "ocr_pages": []}
_SCAN = "results/ICICIBANK-2024-06-30.pdf"


@pytest.mark.parametrize(
    ("entry", "built", "documents"),
    [
        pytest.param(_OCR_ENTRY, True, [_SCAN, f"{_SCAN}#rapidocr"], id="a reading after its scan"),
        pytest.param(_OCR_ENTRY, False, [_SCAN], id="a reading not built is not tried"),
        pytest.param(
            _IMAGE_ENTRY, True, [f"{_SCAN}#rapidocr"], id="an image-only scan's reading alone"
        ),
        pytest.param(FLAGGED, True, [_SCAN], id="a text pdf has no reading"),
    ],
)
def test_facts_to_check_try_a_scan_then_its_reading(
    crosscheck_db: sqlite3.Connection, entry: Entry, built: bool, documents: list[str]
) -> None:
    facts = facts_to_check(
        crosscheck_db, ("ICICIBANK",), {_SCAN: entry}, ("ICICIBANK",), built=lambda _: built
    )
    first = facts[0].fact_id
    assert [f.document for f in facts if f.fact_id == first] == documents


def test_crosscheck_keeps_a_readings_agreement_over_its_scans_silence(
    crosscheck_db: sqlite3.Connection, tmp_path: Path
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({_SCAN: _OCR_ENTRY}), encoding="utf-8")
    silent = _document("STANDALONE FINANCIAL RESULTS (₹ in crore)")

    def load(name: str, verify_dir: Path) -> Document:
        return STANDALONE if name.endswith("#rapidocr") else silent

    checked = crosscheck(
        crosscheck_db, ("ICICIBANK",), tmp_path, load, lambda _: True, ("ICICIBANK",)
    )
    agreed = {c.document for c in checked if c.outcome == "agree"}
    assert agreed == {f"{_SCAN}#rapidocr"}, [(c.concept, c.outcome, c.document) for c in checked]


_QUEUE_MANIFEST: dict[str, Entry] = {
    "results/WIPRO-2025-03-31.pdf": {**_OCR_ENTRY},
    "results/WIPRO-2025-06-30.pdf": {**_OCR_ENTRY},
    "results/ITC-2025-03-31.pdf": {**_OCR_ENTRY},
    "results/ITC-2025-06-30.pdf": {**FLAGGED},
}


@pytest.mark.parametrize(
    ("documents", "tickers", "built", "todo", "unknown"),
    [
        pytest.param(
            [],
            [],
            set(),
            ["ITC-2025-03-31", "WIPRO-2025-03-31", "WIPRO-2025-06-30"],
            [],
            id="every scan when none is named",
        ),
        pytest.param(
            [], ["WIPRO"], set(), ["WIPRO-2025-03-31", "WIPRO-2025-06-30"], [], id="a company's"
        ),
        pytest.param(
            ["results/ITC-2025-03-31.pdf"], [], set(), ["ITC-2025-03-31"], [], id="a document"
        ),
        pytest.param(
            [], ["WIPRO"], {"WIPRO-2025-03-31"}, ["WIPRO-2025-06-30"], [], id="built ones skipped"
        ),
        pytest.param(
            ["results/ITC-2025-06-30.pdf"],
            [],
            set(),
            [],
            ["results/ITC-2025-06-30.pdf"],
            id="a text pdf asked for has none",
        ),
    ],
)
def test_reocr_queue(
    documents: list[str], tickers: list[str], built: set[str], todo: list[str], unknown: list[str]
) -> None:
    def is_built(reading: str) -> bool:
        return Path(reading.removesuffix("#rapidocr")).stem in built

    got, missing = cli.reocr_queue(_QUEUE_MANIFEST, documents, tickers, is_built)
    assert [Path(r.removesuffix("#rapidocr")).stem for r in got] == todo, got
    assert missing == unknown


@pytest.mark.parametrize(
    ("code", "rss", "outcome"),
    [
        pytest.param("pass", 0.5, "ok", id="a run that ends"),
        pytest.param("raise SystemExit(3)", 0.5, "failed exit 3", id="a run that fails"),
        pytest.param("import time; time.sleep(30)", 99.0, "stopped over 10 GB", id="one too big"),
    ],
)
def test_a_guarded_run_is_stopped_past_its_memory(code: str, rss: float, outcome: str) -> None:
    got, peak = cli.guarded([sys.executable, "-c", code], 10.0, 0.05, lambda _: rss)
    assert got == outcome, (got, peak)


def test_a_process_tree_s_memory_is_counted() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.5)
        own = psutil.Process(os.getpid()).memory_info().rss / 2**30
        assert cli.tree_rss_gb(os.getpid()) > own, "a child's memory is counted"
    finally:
        child.kill()
        child.wait()
    assert cli.tree_rss_gb(2**22 + 7) == 0.0


def test_the_command_names_readings_not_yet_built(
    crosscheck_db: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = Path(crosscheck_db.execute("PRAGMA database_list").fetchone()[2])
    crosscheck_db.close()
    monkeypatch.setattr(cli, "FACTS_DB", db)
    monkeypatch.setattr(cli, "CROSSCHECK_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli, "crosscheck", lambda c, t: [])
    monkeypatch.setattr(cli, "read_manifest", lambda: {"results/WIPRO-2025-03-31.pdf": _OCR_ENTRY})
    monkeypatch.setattr(cli, "reocr_built", lambda _: False)
    cli.cross_check(NOW, write=False, decisions=tmp_path / "none.jsonl")
    assert "1 scans have no re-OCR'd reading yet" in capsys.readouterr().out
