"""The XBRL cross-check: printed numbers read, column periods, scale and basis, each fact's outcome,
the tier it moves to, the sample for a person and the report."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest

from aigent.facts import __main__ as cli
from aigent.facts.crosscheck import (
    Checked,
    Column,
    Fact,
    agrees,
    apply,
    check_fact,
    column_periods,
    crosscheck,
    document_scale,
    facts_to_check,
    label_share,
    number,
    review_sample,
    sheets,
    write_report,
)
from aigent.facts.store import FilingRow, Writer, connect, sync_xbrl_facts, upsert_filing
from aigent.facts.xbrl import Fact as XbrlFact
from aigent.retrieval.chunk import Document, Table

WRITER = Writer("xbrl-loader@abc1234", "2026-10-22T12:00:00+05:30", "build-1")
Q1 = (date(2024, 4, 1), date(2024, 6, 30))
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


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    """Two ICICI Bank filings, Q1 standalone and consolidated, one HDFC Bank, and a quarter with no
    document."""
    db = connect(tmp_path / "facts.sqlite")
    for ticker, end, basis, facts in (
        (
            "ICICIBANK",
            date(2024, 6, 30),
            "standalone",
            [
                XbrlFact("InterestEarned", "duration", *Q1, None, 389957800000.0, "INR", -5),
                XbrlFact("OtherIncome", "duration", *Q1, None, 99.0, "INR", -5),
            ],
        ),
        (
            "ICICIBANK",
            date(2024, 6, 30),
            "consolidated",
            [XbrlFact("InterestEarned", "duration", *Q1, None, 445816500000.0, "INR", -5)],
        ),
        (
            "ICICIBANK",
            date(2024, 9, 30),
            "standalone",
            [
                XbrlFact(
                    "InterestEarned",
                    "duration",
                    date(2024, 7, 1),
                    date(2024, 9, 30),
                    None,
                    1.0,
                    "INR",
                    -5,
                )
            ],
        ),
        (
            "HDFCBANK",
            date(2024, 6, 30),
            "standalone",
            [XbrlFact("InterestEarned", "duration", *Q1, None, 1.0, "INR", -5)],
        ),
    ):
        fid = upsert_filing(
            db, FilingRow(ticker, 2025, end, basis, "xbrl", f"{ticker}{end}{basis}", "s", "t")
        )
        sync_xbrl_facts(db, fid, ticker, basis, facts, WRITER)
    db.execute("INSERT INTO concepts (concept, label, kind) VALUES ('X', 'X', 'xbrl')")
    db.execute(
        "INSERT INTO concept_labels (concept, label) VALUES ('InterestEarned', 'Interest income')"
    )
    db.commit()
    return db


MANIFEST = {"results/ICICIBANK-2024-06-30.pdf": {}, "results/HDFCBANK-2024-06-30.pdf": {}}


def test_facts_to_check_are_the_companies_named_with_a_document(
    conn: sqlite3.Connection,
) -> None:
    facts = facts_to_check(conn, ("ICICIBANK",), MANIFEST)
    assert [(f.basis, f.concept, f.document) for f in facts] == [
        ("consolidated", "InterestEarned", "results/ICICIBANK-2024-06-30.pdf"),
        ("standalone", "InterestEarned", "results/ICICIBANK-2024-06-30.pdf"),
        ("standalone", "OtherIncome", "results/ICICIBANK-2024-06-30.pdf"),
    ]
    assert facts[0].labels[1:] == ("Interest income",)


def test_facts_to_check_skip_a_fact_not_from_xbrl(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE facts SET tier = 'pdf-cell-ai' WHERE concept = 'OtherIncome'")
    facts = facts_to_check(conn, ("ICICIBANK",), MANIFEST)
    assert "OtherIncome" not in {f.concept for f in facts}


def test_crosscheck_reads_each_document_once(conn: sqlite3.Connection, tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    loaded: list[str] = []

    def load(name: str, verify_dir: Path) -> Document:
        loaded.append(name)
        return STANDALONE

    checked = crosscheck(conn, ("ICICIBANK",), tmp_path, load)
    assert loaded == ["results/ICICIBANK-2024-06-30.pdf"]
    assert [(c.basis, c.concept, c.outcome) for c in checked] == [
        ("consolidated", "InterestEarned", "unmatched"),
        ("standalone", "InterestEarned", "agree"),
        ("standalone", "OtherIncome", "differ"),
    ]


def _checked(fact_id: int, outcome: str, tier: str = "xbrl-ai") -> Checked:
    return Checked(
        fact_id,
        "ICICIBANK",
        "standalone",
        "InterestEarned",
        None,
        "2024-04-01",
        "2024-06-30",
        1.0,
        "INR",
        tier,
        "results/ICICIBANK-2024-06-30.pdf",
        outcome,  # type: ignore[arg-type]
        3,
        "38,995.78",
        "Interest earned",
    )


def test_apply_makes_only_agreements_xbrl_pdf(conn: sqlite3.Connection) -> None:
    ids = [r[0] for r in conn.execute("SELECT id FROM facts ORDER BY id")]
    moved = apply(conn, [_checked(ids[0], "agree"), _checked(ids[1], "differ")], NOW)
    rows = conn.execute(
        "SELECT tier, verified_by, verified_at, printed, page FROM facts ORDER BY id"
    ).fetchall()
    assert moved == 1
    assert rows[0] == (
        "xbrl-pdf",
        "pdf:results/ICICIBANK-2024-06-30.pdf",
        "2026-11-01T09:00:00",
        "38,995.78",
        3,
    )
    assert rows[1] == ("xbrl-ai", None, None, None, None)


def test_apply_leaves_a_persons_check_and_counts_nothing_twice(conn: sqlite3.Connection) -> None:
    ids = [r[0] for r in conn.execute("SELECT id FROM facts ORDER BY id")]
    conn.execute(
        "UPDATE facts SET tier = 'xbrl', verified_by = 'vedant', verified_at = 't' WHERE id = ?",
        (ids[1],),
    )
    first = apply(conn, [_checked(ids[0], "agree"), _checked(ids[1], "agree", "xbrl-ai")], NOW)
    again = apply(conn, [_checked(ids[0], "agree")], NOW)
    assert (first, again) == (1, 0)
    assert conn.execute(
        "SELECT tier, verified_by FROM facts WHERE id = ?", (ids[1],)
    ).fetchone() == (
        "xbrl",
        "vedant",
    )


def test_the_review_sample_is_seeded_agreements_only() -> None:
    checked = [_checked(i, "agree" if i % 2 else "differ") for i in range(1, 41)]
    first, again = review_sample(checked, 5, 7), review_sample(checked, 5, 7)
    assert first == again and len(first) == 5
    assert all(c.outcome == "agree" for c in first)
    assert len(review_sample(checked, 100, 7)) == 20


def test_the_report_counts_and_queues_every_fact_not_agreed(tmp_path: Path) -> None:
    checked = [_checked(1, "agree"), _checked(2, "differ"), _checked(3, "unmatched")]
    path = write_report(checked, [checked[0]], {"run": "now"}, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    queued, sampled = text.split("## Review queue: sampled agreements")
    assert "| ICICIBANK | standalone | 1 | 1 | 1 |" in text
    assert "| 2 |" in queued and "| 3 |" in queued and "| 1 |" not in queued.split("## Review")[-1]
    assert "| 1 |" in sampled
    rows = path.with_suffix(".rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(r)["outcome"] for r in rows] == ["agree", "differ", "unmatched"]


@pytest.mark.parametrize(("write", "tier"), [(False, "xbrl-ai"), (True, "xbrl-pdf")])
def test_the_command_records_agreements_only_when_told(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write: bool,
    tier: str,
) -> None:
    db = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    first = conn.execute("SELECT min(id) FROM facts").fetchone()[0]
    conn.close()
    monkeypatch.setattr(cli, "FACTS_DB", db)
    monkeypatch.setattr(cli, "CROSSCHECK_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli, "crosscheck", lambda c, t: [_checked(first, "agree")])
    cli.cross_check(NOW, write=write)
    after = sqlite3.connect(db).execute("SELECT tier FROM facts WHERE id = ?", (first,)).fetchone()
    assert after == (tier,)
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
