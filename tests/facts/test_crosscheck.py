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
from aigent.facts import crosscheck as crosscheck_module
from aigent.facts.crosscheck import (
    CROSSCHECK_FINGERPRINT,
    Column,
    Fact,
    RowRead,
    agrees,
    apply,
    check_fact,
    crosscheck,
    documents_for,
    facts_to_check,
    fingerprint,
    recheck,
    sheets,
)
from aigent.facts.documents import Entry
from aigent.facts.store import DerivedRow, FilingRow, sync_xbrl_facts, upsert_filing, write_derived
from aigent.facts.xbrl import Fact as XbrlFact
from aigent.retrieval.chunk import Document, Table

from .conftest import (
    CROSSCHECK_HEADER,
    CROSSCHECK_Q1,
    CROSSCHECK_RESULTS,
    CROSSCHECK_WRITER,
    checked_row,
    crosscheck_document,
)

Q1 = CROSSCHECK_Q1
NOW = datetime(2026, 11, 1, 9, 0)

_HEADER = CROSSCHECK_HEADER
_RESULTS = CROSSCHECK_RESULTS
_document = crosscheck_document


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


@pytest.mark.parametrize(
    ("heading", "basis"),
    [
        pytest.param("STATEMENT OF CONSOUDATED RESULTS", "consolidated", id="OCR's CONSOUDATED"),
        pytest.param("STANDALQNE RESULTS", "standalone", id="OCR's STANDALQNE"),
        pytest.param("Basis of consolidation", "standalone", id="consolidation is no basis"),
    ],
)
def test_a_basis_is_read_through_ocr_misspellings(heading: str, basis: str) -> None:
    document = _document("STANDALONE RESULTS (₹ in crore)", _RESULTS, heading, _RESULTS)
    assert sheets(document)[1].basis == basis


@pytest.mark.parametrize(
    ("heading", "framework"),
    [
        pytest.param("Consolidated results under IFRS", "ifrs", id="IFRS declared"),
        pytest.param(
            "Prepared in compliance with International Financial Reporting Standards",
            "ifrs",
            id="IFRS declared in full",
        ),
        pytest.param("Results under Ind AS", "ind as", id="Ind AS declared"),
        pytest.param("Segment results", "ind as", id="none named: the table before's"),
        pytest.param("Ind AS 116, akin to IFRS 16, applied", "ind as", id="a standard cited"),
        pytest.param(
            "IFRS S2 climate disclosures; the IFRS Foundation",
            "ind as",
            id="IFRS named but not as the statements' framework",
        ),
        pytest.param("Under IFRS above; results under lndAS follow", "ind as", id="OCR's lndAS"),
        pytest.param("Under Ind AS above; results under IFRS follow", "ifrs", id="the last named"),
    ],
)
def test_a_sheet_reads_its_accounting_framework(heading: str, framework: str) -> None:
    document = _document("STANDALONE RESULTS under Ind AS", _RESULTS, heading, _RESULTS)
    assert sheets(document)[1].framework == framework


@pytest.mark.parametrize(
    ("heading", "outcome"),
    [
        pytest.param("STANDALONE RESULTS under IFRS", "unmatched", id="an IFRS table"),
        pytest.param("STANDALONE RESULTS under Ind AS", "agree", id="an Ind AS table"),
        pytest.param("STANDALONE RESULTS", "agree", id="a table naming no framework"),
    ],
)
def test_an_xbrl_fact_is_checked_only_against_its_framework(heading: str, outcome: str) -> None:
    checked = check_fact(_fact(), sheets(_document(f"{heading} (₹ in crore)", _RESULTS)))
    assert checked.outcome == outcome, checked


_QUARTER_AND_NINE = {1: Column(date(2024, 12, 31), None), 2: Column(date(2024, 12, 31), 9)}


@pytest.mark.parametrize(
    ("columns", "period", "within"),
    [
        pytest.param(
            _QUARTER_AND_NINE,
            (date(2024, 10, 1), date(2024, 12, 31)),
            True,
            id="the one column of no length a quarter could be",
        ),
        pytest.param(
            {**_QUARTER_AND_NINE, 2: Column(date(2024, 12, 31), 3)},
            (date(2024, 10, 1), date(2024, 12, 31)),
            False,
            id="not when another column ending that day states the quarter",
        ),
        pytest.param(
            {**_QUARTER_AND_NINE, 3: Column(date(2024, 12, 31), None)},
            (date(2024, 10, 1), date(2024, 12, 31)),
            False,
            id="nor when two of no length end that day",
        ),
    ],
)
def test_a_column_of_no_length_is_a_periods_when_no_other_could_be(
    columns: dict[int, Column], period: tuple[date, date], within: bool
) -> None:
    fact = _fact(period=period)
    assert crosscheck_module._in_period(fact, columns[1], columns) is within


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


_TOTAL = ("", "", "1,234", "1", "1", "1")
_SHARES = ("", "Equity share capital", "737", "1", "1", "1")
_RESERVES = ("", "Other equity", "497", "1", "1", "1")


@pytest.mark.parametrize(
    ("block", "agreed"),
    [
        pytest.param((_SHARES, _RESERVES, _TOTAL), True, id="an unlabelled last row"),
        pytest.param((_TOTAL, _SHARES, _RESERVES), True, id="an unlabelled first row"),
        pytest.param((_SHARES, _TOTAL, _RESERVES), True, id="a subtotal with figures following it"),
        pytest.param(
            (_SHARES, ("", "", "1", "1", "1", "1"), _RESERVES, _TOTAL),
            False,
            id="a second unlabelled row after the subtotal",
        ),
        pytest.param(
            (_SHARES, _RESERVES, ("", "Other reserves", "1,234", "1", "1", "1")),
            False,
            id="a labelled last row keeps its label",
        ),
    ],
)
def test_a_headings_first_row_or_closing_subtotal_reads_as_it(
    block: tuple[tuple[str, ...], ...], agreed: bool
) -> None:
    scaled = ("", "STANDALONE BALANCE SHEET (₹ in crore)", "", "", "", "")
    first = ("", "Revenue", "1", "2", "3", "4")
    rows = (scaled, *_HEADER, first, ("", "Equity", "", "", "", ""), *block)
    read = sheets(_pages((1, rows)))
    fact = _fact("Equity", 12_340_000_000.0, ("Equity",), period=(None, date(2024, 6, 30)))
    assert (check_fact(fact, read).outcome == "agree") is agreed, check_fact(fact, read)


@pytest.mark.parametrize(
    ("rows", "label", "agreed"),
    [
        pytest.param(
            (
                ("", "VIII Tax expense:", "", "", "", ""),
                ("", "Deferred tax", "1", "1", "1", "1"),
                ("IX", "Profit before regulatory movements", "2", "2", "2", "2"),
                ("XI", "Profit for the period", "3", "3", "3", "3"),
                ("", "", "1,234", "1", "1", "1"),
            ),
            "Profit for the period",
            True,
            id="a subtotal after a numbered line takes that line, not a heading above",
        ),
        pytest.param(
            (
                ("", "VIII Tax expense:", "", "", "", ""),
                ("", "Deferred tax", "1", "1", "1", "1"),
                ("XI", "Profit for the period", "3", "3", "3", "3"),
                ("", "", "1,234", "1", "1", "1"),
            ),
            "Tax expense",
            False,
            id="the heading above a numbered line names nothing after it",
        ),
    ],
)
def test_a_numbered_main_line_heads_its_own_block(
    rows: tuple[tuple[str, ...], ...], label: str, agreed: bool
) -> None:
    scaled = ("", "STANDALONE RESULTS (₹ in crore)", "", "", "", "")
    read = sheets(_pages((1, (scaled, *_HEADER, ("", "Revenue", "1", "2", "3", "4"), *rows))))
    fact = _fact("X", 12_340_000_000.0, (label,))
    assert (check_fact(fact, read).outcome == "agree") is agreed, check_fact(fact, read)


@pytest.mark.parametrize(
    ("rows", "read"),
    [
        pytest.param(
            (
                ("A 1. II. B.", "Segment Revenue", "", "", "", ""),
                ("A 1. II. B.", "(a) Commercial Vehicle", "1", "1", "1", "1"),
                ("A 1. II. B.", "(b) Passenger Vehicle", "1", "1", "1", "1"),
            ),
            (("heading", 1), ("figures", 1), ("figures", 1)),
            id="a spanning cell copied down a segment table makes no main lines",
        ),
        pytest.param(
            (
                ("B.", "Cash flow from investing activities", "1", "1", "1", "1"),
                ("B.", "Purchase of property", "1", "1", "1", "1"),
                ("B.", "Dividend received", "1", "1", "1", "1"),
            ),
            (("main line", 1), ("figures", 1), ("figures", 1)),
            id="a copied main line heads the block its copies sit in",
        ),
        pytest.param(
            (
                ("B.", "Cash flow from investing activities", "1", "1", "1", "1"),
                ("B.", "Cash flow from financing activities", "1", "1", "1", "1"),
            ),
            (("main line", 1), ("main line", 2)),
            id="the same numeral on two rows is two main lines",
        ),
    ],
)
def test_a_spanning_cells_copies_read_as_its_rows(
    rows: tuple[tuple[str, ...], ...], read: tuple[tuple[str, int], ...]
) -> None:
    body = (("", "Revenue", "1", "2", "3", "4"), *rows)
    [sheet] = sheets(_document("STANDALONE (₹ in crore)", (*_HEADER, *body)))
    got = tuple((x.kind, x.block) for x in sheet.rows[1:])
    assert got == read, sheet.rows


@pytest.mark.parametrize(
    ("rows", "agreed"),
    [
        pytest.param(
            (
                ("XIII", "Earnings per equity share Basic", "15.73", "1", "1", "1"),
                _eps_row("Diluted"),
            ),
            True,
            id="a diluted row under a numbered EPS line",
        ),
        pytest.param(
            (
                _EPS_HEADING,
                ("a)", "Basic (in ₹)", "1", "1", "1", "1"),
                ("b)", "Diluted", "15.73", "1", "1", "1"),
            ),
            True,
            id="lettered sub-rows keep the EPS heading",
        ),
    ],
)
def test_eps_rows_read_with_their_main_line(
    rows: tuple[tuple[str, ...], ...], agreed: bool
) -> None:
    read = sheets(_pages((1, (*_HEADER, ("1", "Interest earned", "1", "2", "3", "4"), *rows))))
    fact = _fact("EPS", 15.73, (_DILUTED_EPS,), unit="INRPerShare", decimals=2)
    assert (check_fact(fact, read).outcome == "agree") is agreed, check_fact(fact, read)


def test_a_heading_carries_into_the_statement_it_continues() -> None:
    first = (*_HEADER, ("1", "Interest earned", "1.00", "2.00", "3.00", "4.00"), _EPS_HEADING)
    read = sheets(_pages((1, first), (2, (_eps_row("Basic (in ₹)"),))))
    fact = _fact("EPS", 15.73, (_TOTAL_EPS,), unit="INRPerShare", decimals=2)
    assert check_fact(fact, read).outcome == "agree", read[-1].cells[:1]


def test_a_rows_serial_number_is_not_a_figure() -> None:
    [sheet] = sheets(STANDALONE)
    assert all(c.column != 0 for c in sheet.cells)
    assert sheet.cells[0].label == "Interest earned"


@pytest.mark.parametrize(
    ("note", "heading"),
    [
        pytest.param("14", "EARNINGS PER SHARE", id="a note number heads no figures"),
        pytest.param("4.1", "EARNINGS PER SHARE", id="a numbered sub-note too"),
        pytest.param(
            "15.15",
            "Particulars Note 30.06.2024 31.03.2024",
            id="a figure of two places left of the dates stays one",
        ),
    ],
)
def test_a_note_reference_left_of_the_dates_is_not_a_figure(note: str, heading: str) -> None:
    rows = (
        ("", "Particulars", "Note", "30.06.2024", "31.03.2024"),
        ("", "Revenue", "", "100", "90"),
        ("", "EARNINGS PER SHARE", note, "", ""),
        ("(a)", "Basic", "", "4.47", "4.06"),
    )
    [sheet] = sheets(_document("STANDALONE (₹ in crore)", rows))
    basic = next(c for c in sheet.cells if c.printed == "4.47")
    assert basic.heading == heading, sheet.cells
    assert any(c.column == 2 for c in sheet.cells) is (note == "15.15"), sheet.cells


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
        pytest.param(
            _fact(concept="InterestPaidClassifiedAsFinancingActivities", value=462500000.0),
            "(46.25)",
            1e7,
            True,
            id="a payment filed positive, printed in brackets",
        ),
        pytest.param(
            _fact(concept="AdjustmentsForInterestIncome", value=8743300000.0),
            "(874.33)",
            1e7,
            True,
            id="interest income deducted in the cash flow",
        ),
        pytest.param(
            _fact(concept="ProfitLossForPeriod", value=462500000.0),
            "(46.25)",
            1e7,
            False,
            id="a loss is no profit",
        ),
        pytest.param(
            _fact(concept="InterestPaidClassifiedAsFinancingActivities", value=462500000.0),
            "(46.26)",
            1e7,
            False,
            id="a payment still to the last digit",
        ),
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


_TCI = ("Total comprehensive income for the period",)
_TCI_FIGURE = ("1,000.00", "1", "1", "1")


def _row(n: str, label: str, figures: tuple[str, ...] = _TCI_FIGURE) -> tuple[str, ...]:
    return (n, label, *figures)


@pytest.mark.parametrize(
    ("tables", "agreed"),
    [
        pytest.param(
            [
                (1, (*_HEADER, _row("1", "Total comprehensive income from continuing operations"))),
                (1, (*_HEADER, _row("2", "Total comprehensive income for the period"))),
            ],
            (1, "Total comprehensive income for the period", 1.0),
            id="a full label later beats a partial one first",
        ),
        pytest.param(
            [
                (1, (*_HEADER, _row("1", "Total comprehensive income for the period"))),
                (1, (*_HEADER, _row("2", "Total comprehensive income for the period (net)"))),
            ],
            (0, "Total comprehensive income for the period", 1.0),
            id="on a tie the first is kept",
        ),
        pytest.param(
            [
                (1, (*_HEADER, _row("1", "Statement line", ("1,000.00", "", "", "")))),
                (1, (*_HEADER, _row("2", "Total comprehensive income for the period"))),
            ],
            (1, "Total comprehensive income for the period", 1.0),
            id="a labelled row beats a figure agreeing only as distinctive",
        ),
        pytest.param(
            [
                (1, (*_HEADER, _row("1", "Total comprehensive income from continuing operations"))),
                (2, (*_HEADER, _row("2", "Total comprehensive income for the period"))),
            ],
            (1, "Total comprehensive income for the period", 1.0),
            id="a better row on a later page of the document",
        ),
        pytest.param(
            [
                (1, (*_HEADER, _row("1", "Total comprehensive income from continuing operations"))),
                (
                    1,
                    (
                        *_HEADER,
                        _row(
                            "2",
                            "Total comprehensive income for the period",
                            ("2,000.00", "1", "1", "1"),
                        ),
                    ),
                ),
            ],
            (0, "Total comprehensive income from continuing operations", 0.75),
            id="a better label printing another figure does not agree",
        ),
    ],
)
def test_the_best_labelled_agreeing_cell_is_taken(
    tables: list[tuple[int, tuple[tuple[str, ...], ...]]], agreed: tuple[int, str, float]
) -> None:
    fact = _fact("ComprehensiveIncomeForThePeriod", 10000000000.0, _TCI)
    read = [replace(sheet, scale=1e7) for sheet in sheets(_pages(*tables))]
    checked = check_fact(fact, read)
    assert checked.outcome == "agree", checked
    assert (checked.sheet, checked.row_label, checked.share) == agreed, checked


def test_a_fact_records_where_it_agreed() -> None:
    checked = check_fact(_fact(), sheets(STANDALONE))
    where = (checked.page, checked.row_label, checked.sheet, checked.row, checked.column)
    assert where == (1, "Interest earned", 0, 2, 2), checked
    assert (checked.heading, checked.share) == (" ".join(_HEADER[-1]), 1.0), checked


_EPS_UNDER_HEADING = _pages((1, (*_HEADER, _EPS_HEADING, _eps_row("a) Basic (in ₹)"))))


def test_a_fact_not_agreed_names_the_veto_that_refused_an_equal_figure() -> None:
    rows = (*_HEADER, ("1", "Profit before exceptional items", "12.50", "", "", ""))
    fact = _fact(concept="ExceptionalItems", labels=("Exceptional items",), value=125000000.0)
    checked = check_fact(fact, sheets(_document("STANDALONE (₹ in crore)", rows)))
    assert (checked.outcome, checked.veto) == (
        "unmatched",
        "row turns: Profit before exceptional items",
    ), checked


def test_an_agreed_fact_names_no_veto() -> None:
    checked = check_fact(_fact(), sheets(STANDALONE))
    assert (checked.outcome, checked.veto) == ("agree", None), checked


@pytest.mark.parametrize(
    ("fact", "document", "outcome", "rule"),
    [
        pytest.param(_fact(), STANDALONE, "agree", "label", id="a labelled distinct figure"),
        pytest.param(
            _fact("OtherIncome", 70019200000.0, ("Income not from interest",)),
            STANDALONE,
            "agree",
            "distinct",
            id="a distinctive figure in an unlabelled row",
        ),
        pytest.param(
            _fact("EPS", 15.73, (_TOTAL_EPS,), unit="INRPerShare", decimals=2),
            _EPS_UNDER_HEADING,
            "agree",
            "per-share",
            id="a short figure under its per-share heading",
        ),
        pytest.param(
            _fact(value=390000000000.0), STANDALONE, "differ", None, id="a difference has no rule"
        ),
    ],
)
def test_an_agreement_records_the_rule_it_agreed_by(
    fact: Fact, document: Document, outcome: str, rule: str | None
) -> None:
    checked = check_fact(fact, sheets(document))
    assert (checked.outcome, checked.rule) == (outcome, rule), checked


def test_a_sheet_records_how_it_read_each_row() -> None:
    rows = (
        *_HEADER,
        ("", "Revenue", "1", "2", "3", "4"),
        ("", "Equity", "", "", "", ""),
        _SHARES,
        _RESERVES,
        _TOTAL,
        ("IX", "Tax expense", "5", "6", "7", "8"),
    )
    [sheet] = sheets(_document("STANDALONE (₹ in crore)", rows))
    top = " ".join(_HEADER[-1])
    assert sheet.header == 2
    assert sheet.rows == (
        RowRead(2, "figures", "Revenue", "Revenue", top, 0),
        RowRead(3, "heading", "Equity", "", "Equity", 1),
        RowRead(4, "figures", "Equity share capital", "Equity share capital", "Equity", 1),
        RowRead(5, "figures", "Other equity", "Other equity", "Equity", 1),
        RowRead(6, "figures", "", "Equity", "Equity", 1),
        RowRead(7, "main line", "IX Tax expense", "IX Tax expense", "IX Tax expense", 2),
    ), sheet.rows


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
        "SELECT tier, verified_by, verified_at, printed, page, verified_with FROM facts ORDER BY id"
    ).fetchall()
    assert moved == 1
    assert rows[0] == (
        "xbrl-filed",
        "nse-pdf:results/ICICIBANK-2024-06-30.pdf",
        "2026-11-01T09:00:00",
        "38,995.78",
        3,
        CROSSCHECK_FINGERPRINT,
    )
    assert rows[1] == ("xbrl-ai", None, None, None, None, None)


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


FILED_BY = "nse-pdf:results/ICICIBANK-2024-06-30.pdf"
RECHECK_ORIGINS = {**ORIGINS, "results/X.htm": "sec-6k-html"}
STAMPED = (FILED_BY, "t0", "38,995.78", 3)
AFRESH = ("xbrl-filed", FILED_BY, "2026-11-01T09:00:00", "38,995.78", 3, "now")
DEMOTED = ("xbrl-ai", None, None, None, None, None)


def _filed(db: sqlite3.Connection, fid: int, reading: str | None) -> None:
    db.execute(
        """UPDATE facts SET tier = 'xbrl-filed', verified_by = ?, verified_at = ?, printed = ?,
                  page = ?, verified_with = ? WHERE id = ?""",
        (*STAMPED, reading, fid),
    )


def _stamp(db: sqlite3.Connection, fid: int) -> tuple[object, ...]:
    return db.execute(
        """SELECT tier, verified_by, verified_at, printed, page, verified_with FROM facts
           WHERE id = ?""",
        (fid,),
    ).fetchone()


@pytest.mark.parametrize(
    ("reading", "outcome", "moved", "after", "counts"),
    [
        pytest.param("old", "agree", {}, AFRESH, (1, 0), id="a stale stamp still agreeing"),
        pytest.param(None, "agree", {}, AFRESH, (1, 0), id="a check from before stamps"),
        pytest.param(
            "old",
            "agree",
            {"document": "results/X.htm", "page": 7, "printed": "1.0"},
            ("xbrl-filed", "sec-6k-html:results/X.htm", "2026-11-01T09:00:00", "1.0", 7, "now"),
            (1, 0),
            id="an agreement moved to another copy and page",
        ),
        pytest.param("old", "differ", {}, DEMOTED, (0, 1), id="a stale stamp now differing"),
        pytest.param("old", "unmatched", {}, DEMOTED, (0, 1), id="a stale stamp now unmatched"),
        pytest.param("old", None, {}, DEMOTED, (0, 1), id="a stale stamp no longer checked"),
        pytest.param(
            "now",
            "differ",
            {},
            ("xbrl-filed", *STAMPED, "now"),
            (0, 0),
            id="a current stamp is left alone",
        ),
    ],
)
def test_recheck_stamps_afresh_or_demotes_a_stale_check(
    crosscheck_db: sqlite3.Connection,
    reading: str | None,
    outcome: str | None,
    moved: dict[str, object],
    after: tuple[object, ...],
    counts: tuple[int, int],
) -> None:
    first = crosscheck_db.execute("SELECT min(id) FROM facts").fetchone()[0]
    _filed(crosscheck_db, first, reading)
    checked = [] if outcome is None else [replace(checked_row(first, outcome), **moved)]
    done = recheck(crosscheck_db, checked, NOW, RECHECK_ORIGINS, ("ICICIBANK",), "now")
    assert (done.restamped, done.demoted) == counts, done
    assert _stamp(crosscheck_db, first) == after


def test_recheck_leaves_a_persons_check_and_another_company_alone(
    crosscheck_db: sqlite3.Connection,
) -> None:
    ids = [r[0] for r in crosscheck_db.execute("SELECT id FROM facts ORDER BY id")]
    crosscheck_db.execute(
        "UPDATE facts SET tier = 'xbrl', verified_by = 'vedant', verified_at = 't' WHERE id = ?",
        (ids[1],),
    )
    _filed(crosscheck_db, ids[-1], "old")  # HDFC Bank's
    before = [_stamp(crosscheck_db, i) for i in (ids[1], ids[-1])]
    checked = [checked_row(i, "differ") for i in (ids[1], ids[-1])]
    done = recheck(crosscheck_db, checked, NOW, RECHECK_ORIGINS, ("ICICIBANK",), "now")
    assert (done.restamped, done.demoted) == (0, 0), done
    assert [_stamp(crosscheck_db, i) for i in (ids[1], ids[-1])] == before


def test_recheck_writes_nothing_for_a_document_with_no_origin(
    crosscheck_db: sqlite3.Connection,
) -> None:
    ids = [r[0] for r in crosscheck_db.execute("SELECT id FROM facts ORDER BY id")]
    for fid in ids[:2]:
        _filed(crosscheck_db, fid, "old")
    crosscheck_db.commit()
    checked = [
        checked_row(ids[0], "differ"),
        replace(checked_row(ids[1], "agree"), document="results/NOPE.htm"),
    ]
    with pytest.raises(KeyError, match="NOPE"):
        recheck(crosscheck_db, checked, NOW, RECHECK_ORIGINS, ("ICICIBANK",), "now")
    assert [_stamp(crosscheck_db, i)[0] for i in ids[:2]] == ["xbrl-filed", "xbrl-filed"]


@pytest.mark.parametrize(
    ("outcome", "tier", "derived_demoted"),
    [
        pytest.param("agree", "derived", 0, id="an input still agreeing keeps it checked"),
        pytest.param("differ", "derived-ai", 1, id="an input demoted unchecks it"),
    ],
)
def test_recheck_takes_a_derived_facts_check_with_its_inputs(
    crosscheck_db: sqlite3.Connection, outcome: str, tier: str, derived_demoted: int
) -> None:
    first = crosscheck_db.execute("SELECT min(id) FROM facts").fetchone()[0]
    _filed(crosscheck_db, first, "old")
    row = DerivedRow(
        "ICICIBANK", "standalone", "Half", "Half", None, "2024-06-30", None, 1.0, "INR", "{}"
    )
    derived = write_derived(crosscheck_db, row, [first], CROSSCHECK_WRITER, checked_by="vedant")
    done = recheck(
        crosscheck_db, [checked_row(first, outcome)], NOW, ORIGINS, ("ICICIBANK",), "now"
    )
    assert done.derived_demoted == derived_demoted, done
    held = crosscheck_db.execute("SELECT tier FROM facts WHERE id = ?", (derived,)).fetchone()
    assert held == (tier,)


@pytest.mark.parametrize(
    ("parts", "other"),
    [
        pytest.param(("a", "b"), ("a", "c"), id="a changed part"),
        pytest.param(("a", "b"), ("b", "a"), id="parts reordered"),
        pytest.param(("a", "b"), ("a", "b", ""), id="a part added"),
        pytest.param(("a|b",), ("a", "b"), id="parts split differently"),
    ],
)
def test_a_fingerprint_moves_with_any_change_to_its_parts(
    parts: tuple[str, ...], other: tuple[str, ...]
) -> None:
    assert fingerprint(parts) == fingerprint(parts)
    assert fingerprint(parts) != fingerprint(other)


@pytest.mark.parametrize(
    ("stale", "after"),
    [
        pytest.param(False, ("xbrl-filed", "old"), id="without --recheck a stale check stands"),
        pytest.param(True, ("xbrl-ai", None), id="--recheck demotes it"),
    ],
)
def test_the_command_rechecks_stale_checks_only_when_told(
    crosscheck_db: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale: bool,
    after: tuple[str, str | None],
) -> None:
    db = Path(crosscheck_db.execute("PRAGMA database_list").fetchone()[2])
    first = crosscheck_db.execute("SELECT min(id) FROM facts").fetchone()[0]
    _filed(crosscheck_db, first, "old")
    crosscheck_db.commit()
    crosscheck_db.close()
    monkeypatch.setattr(cli, "FACTS_DB", db)
    monkeypatch.setattr(cli, "CROSSCHECK_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli, "crosscheck", lambda c, t: [checked_row(first, "differ")])
    manifest = {"results/ICICIBANK-2024-06-30.pdf": {"source": "nse", "format": "pdf"}}
    monkeypatch.setattr(cli, "read_manifest", lambda: manifest)
    cli.cross_check(NOW, write=False, decisions=tmp_path / "none.jsonl", stale=stale)
    row = (
        sqlite3.connect(db)
        .execute("SELECT tier, verified_with FROM facts WHERE id = ?", (first,))
        .fetchone()
    )
    assert row == after, row
    report = next((tmp_path / "reports").glob("crosscheck-*.md")).read_text(encoding="utf-8")
    assert ("1 made xbrl-ai" in report) is stale


@pytest.mark.parametrize(
    ("argv", "flags"),
    [
        pytest.param([], (False, False), id="a plain run records nothing"),
        pytest.param(["--recheck"], (False, True), id="--recheck"),
        pytest.param(["--write"], (True, False), id="--write"),
        pytest.param(["--recheck", "--write"], (True, True), id="both"),
    ],
)
def test_the_crosscheck_flags_reach_the_command(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], flags: tuple[bool, bool]
) -> None:
    seen: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        cli, "cross_check", lambda now, write, stale=False: seen.append((write, stale))
    )
    cli.main(["crosscheck", *argv])
    assert seen == [flags]


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
