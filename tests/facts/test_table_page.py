"""Table pages: the audit flags, the draw, the kept list, the card data and the page."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from cltrplusf.chunk import Document

from aigent.facts import __main__ as cli
from aigent.facts.crosscheck import sheets
from aigent.facts.documents import Entry
from aigent.facts.headers import Column
from aigent.facts.table_page import (
    FLAGS,
    Audit,
    CardSource,
    Found,
    KeptTable,
    TableList,
    audit,
    build_table_page,
    draw_tables,
    figures_in,
    found_tables,
    is_figure,
    read_table_list,
    table_card,
    unfooted,
    write_table_list,
)

from .conftest import CROSSCHECK_HEADER, crosscheck_document

_HEADER = CROSSCHECK_HEADER
_YEAR = (
    ("", "", "Year ended", "Year ended", "Year ended", "Year ended"),
    ("Sr", "Particulars", "31.03.2024", "31.03.2024", "31.03.2024", "31.03.2024"),
)
_UNDATED = (("Sr", "Particulars", "Opening", "Added", "Used", "Closing"),)


def _audit(header: tuple[tuple[str, ...], ...], *body: tuple[str, ...]) -> Audit:
    document = crosscheck_document("STANDALONE (₹ in crore)", (*header, *body))
    [sheet] = sheets(document)
    return audit(document.tables[0], sheet)


@pytest.mark.parametrize(
    ("word", "figure"),
    [
        ("1,234.56", True),
        ("(12)", True),
        ("[504.43]", True),
        ("9", False),
        ("XII", False),
        ("2024-25", False),
        ("ROU", False),
    ],
)
def test_is_figure(word: str, figure: bool) -> None:
    assert is_figure(word) is figure


def test_figures_in_counts_each_printed_figure_without_brackets() -> None:
    rows = (("Revenue 1,000", "[504.43]"), ("12 XII", "1,000"))
    assert figures_in(rows) == {"1,000": 2, "504.43": 1, "12": 1}


@pytest.mark.parametrize(
    ("rows", "found"),
    [
        pytest.param(
            (("a", "10"), ("b", "20"), ("Total", "30")), [], id="a total of the rows above foots"
        ),
        pytest.param(
            (("a", "10"), ("b", "20"), ("Total", "31")), [(2, 1)], id="a total one out does not"
        ),
        pytest.param((("a", "10"), ("Total", "10")), [(1, 1)], id="one row above is no sum"),
        pytest.param((("a", "10"), ("b", "20"), ("Total", "-")), [], id="a nil total is skipped"),
        pytest.param((("a", "10"), ("b", "20"), ("Net", "31")), [], id="a row naming no total"),
        pytest.param(
            (("Total", "30"), ("a", "10"), ("b", "20")),
            [(0, 1)],
            id="a total with nothing above it",
        ),
    ],
)
def test_unfooted(rows: tuple[tuple[str, ...], ...], found: list[tuple[int, int]]) -> None:
    assert unfooted(rows, 0) == found


@pytest.mark.parametrize(
    ("header", "row", "flag", "cells"),
    [
        pytest.param(
            _HEADER,
            ("1", "Interest earned", "38,995.78 37,948.36", "1.00", "2.00", "3.00"),
            "merged figures",
            [(2, 2)],
            id="two figures in one cell",
        ),
        pytest.param(
            _HEADER,
            ("1", "Interest", "earned 38,995.78", "1.00", "2.00", "3.00"),
            "text in a figure column",
            [(2, 2)],
            id="label words in a figure cell",
        ),
        pytest.param(
            _YEAR,
            ("1", "Interest earned", "1.00", "2.00", "3.00", "4.00"),
            "columns share a period",
            [2, 3, 4, 5],
            id="four columns given one year",
        ),
        pytest.param(
            _UNDATED,
            ("1", "Interest earned", "1.00", "2.00", "3.00", "4.00"),
            "no period read",
            [2, 3, 4, 5],
            id="columns with no dates",
        ),
    ],
)
def test_audit_flags_each_fault(
    header: tuple[tuple[str, ...], ...], row: tuple[str, ...], flag: str, cells: list[object]
) -> None:
    found = _audit(header, row)
    by_flag = dict(
        zip(
            FLAGS,
            (found.merged, found.spilled, found.shared, found.unread, found.unfooted),
            strict=True,
        )
    )
    assert list(by_flag[flag]) == cells, found
    assert found.flagged()
    assert [n for f, n in found.counts().items() if f != flag] == [0] * (len(FLAGS) - 1), found


@pytest.mark.parametrize(
    ("period", "unread"),
    [
        pytest.param(Column(date(2024, 6, 30), 3), (), id="a dated column"),
        pytest.param(Column(None, 3), (2,), id="a column whose end was not read"),
    ],
)
def test_a_column_is_unread_without_its_end(period: Column, unread: tuple[int, ...]) -> None:
    document = crosscheck_document("STANDALONE (₹ in crore)", (*_HEADER, _MERGED))
    [sheet] = sheets(document)
    columns = {**sheet.columns, 2: period}
    assert audit(document.tables[0], replace(sheet, columns=columns)).unread == unread


def test_a_clean_table_has_no_flags() -> None:
    found = _audit(_HEADER, ("1", "Interest earned", "38,995.78", "1.00", "2.00", "3.00"))
    assert not found.flagged(), found
    assert list(found.counts()) == list(FLAGS)


def _found(n: int, kind: str, flagged: bool, prefix: str = "d") -> list[Found]:
    return [
        Found(f"{prefix}{kind}{'f' if flagged else 'n'}{i}", i, kind, flagged) for i in range(n)
    ]


_KINDS = ("a", "b")
_POOL = [
    *_found(6, "a", True),
    *_found(6, "a", False),
    *_found(6, "b", True),
    *_found(6, "b", False),
]


def test_draw_tables_takes_half_flagged_spread_over_kinds() -> None:
    page = draw_tables(_POOL, TableList(None, ()), 8, 1, 0.5, _KINDS)
    flagged = {(f.document, f.sheet) for f in _POOL if f.flagged}
    drawn = [(c.document, c.sheet) for c in page.cards if c.pool == "flagged"]
    assert len(drawn) == 4 and set(drawn) <= flagged, page
    assert sorted(c.document[1] for c in page.cards if c.pool == "flagged") == list("aabb")
    assert [c.id for c in sorted(page.cards, key=lambda c: c.id)] == [f"t0{i}" for i in range(1, 9)]
    assert {c.batch for c in page.cards} == {1}


def test_draw_tables_numbers_on_and_never_repeats_a_table() -> None:
    first = draw_tables(_POOL, TableList("u", ()), 8, 1, 0.5, _KINDS)
    second = draw_tables(_POOL, first, 8, 1, 0.5, _KINDS)
    keys = [(c.document, c.sheet) for c in second.cards]
    assert len(keys) == len(set(keys)) == 16
    assert second.url == "u"
    assert {c.batch for c in second.cards[8:]} == {2}
    assert sorted(c.id for c in second.cards[8:])[0] == "t09"


def test_draw_tables_fills_from_any_table_when_few_are_flagged() -> None:
    few = [*_found(1, "a", True), *_found(9, "a", False, "e")]
    page = draw_tables(few, TableList(None, ()), 6, 1, 0.5, _KINDS)
    assert [c.pool for c in page.cards].count("flagged") == 1
    assert len(page.cards) == 6


def test_draw_tables_is_the_same_for_the_same_seed() -> None:
    one = draw_tables(_POOL, TableList(None, ()), 8, 7, 0.5, _KINDS)
    two = draw_tables(_POOL, TableList(None, ()), 8, 7, 0.5, _KINDS)
    assert one == two


def test_a_table_list_reads_back_as_written_sorted_by_id(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    page = TableList(
        "https://x", (KeptTable("t02", "b", 1, 1, "any"), KeptTable("t01", "a", 3, 1, "flagged"))
    )
    write_table_list(path, page)
    assert read_table_list(path) == TableList(
        "https://x", tuple(sorted(page.cards, key=lambda c: c.id))
    )
    assert [json.loads(x.rstrip(","))["id"] for x in path.read_text().splitlines()[1:-1]] == [
        "t01",
        "t02",
    ]
    assert read_table_list(tmp_path / "none.json") == TableList(None, ())


_KEPT = KeptTable("t01", "results/X.pdf", 0, 1, "flagged")
_MERGED = ("1", "Interest earned", "38,995.78 37,948.36", "1.00", "2.00", "3.00")


def _card(docling: tuple[tuple[str, ...], ...] | None, max_rows: int = 160) -> dict[str, object]:
    document = crosscheck_document("STANDALONE (₹ in crore)", (*_HEADER, _MERGED))
    [sheet] = sheets(document)
    source = CardSource("born-digital PDF", "X", docling, None, None)
    return table_card(_KEPT, document.tables[0], sheet, source, max_rows)


def test_a_card_marks_flagged_cells_on_their_rows() -> None:
    card = _card(None)
    rows = card["rows"]
    assert isinstance(rows, list)
    assert [r["marks"] for r in rows] == [{}, {}, {"merged": [2]}]
    assert card["flags"] == {**dict.fromkeys(FLAGS, 0), "merged figures": 1}
    assert card["header"] == 2 and card["page"] == 1


@pytest.mark.parametrize(
    ("docling", "shown", "lost"),
    [
        pytest.param(None, None, [], id="a reading with no Docling rows"),
        pytest.param(
            (*_HEADER, _MERGED), None, [], id="Docling's rows the same as read are not shown"
        ),
        pytest.param(
            (*_HEADER, (*_MERGED[:5], "3.00 99.10")),
            [list(r) for r in (*_HEADER, (*_MERGED[:5], "3.00 99.10"))],
            ["99.10"],
            id="a figure Docling had that the reading lost",
        ),
    ],
)
def test_a_card_shows_docling_rows_and_lost_figures(
    docling: tuple[tuple[str, ...], ...] | None, shown: object, lost: list[str]
) -> None:
    card = _card(docling)
    assert (card["docling"], card["lost"], card["lost_count"]) == (shown, lost, len(lost))


def test_a_card_shows_at_most_its_rows() -> None:
    card = _card(None, max_rows=2)
    rows = card["rows"]
    assert isinstance(rows, list)
    assert (len(rows), card["more_rows"]) == (2, 1)


_TEMPLATE = "{{TITLE}}|{{LEDE}}|{{TAGS}}|{{DATA}}"


def test_the_page_fills_its_markers_and_escapes_script_ends() -> None:
    page = build_table_page([{"id": "t01", "note": "</script> \ufffd"}], _TEMPLATE, ("OK",))
    assert "{{" not in page and '["OK"]' in page and "<\\/script>" in page
    assert (
        "\ufffd" not in page and json.loads(page.split("|", 3)[3])[0]["note"] == "</script> \ufffd"
    )


@pytest.mark.parametrize(
    ("template", "max_mb", "error"),
    [
        pytest.param("{{TITLE}} {{DATA}}", 16.0, "lacks", id="a template missing markers"),
        pytest.param(_TEMPLATE, 0.0, "over", id="a page over the size limit"),
    ],
)
def test_the_page_refuses(template: str, max_mb: float, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        build_table_page([{"id": "t01"}], template, ("OK",), max_mb)


def test_found_tables_lists_every_read_table_with_its_flag() -> None:
    clean = crosscheck_document(
        "STANDALONE (₹ in crore)", (*_HEADER, _MERGED[:2] + ("1.0", "2.0", "3.0", "4.0"))
    )
    merged = crosscheck_document("STANDALONE (₹ in crore)", (*_HEADER, _MERGED))
    docs = {"c": clean, "m": merged}
    got = found_tables(["c", "m"], lambda n: (docs[n].tables, sheets(docs[n])), lambda n: "k")
    assert got == [Found("c", 0, "k", False), Found("m", 0, "k", True)]


_ENTRY: Entry = {
    "source": "nse",
    "format": "pdf",
    "pages": 2,
    "text_less_pages": [],
    "ocr_pages": [],
}


def test_the_command_draws_keeps_and_rebuilds_a_page(tmp_path: Path) -> None:
    manifest = {"results/ITC-2024-06-30.pdf": _ENTRY, "results/HDFCBANK-2024-06-30.pdf": _ENTRY}
    document = crosscheck_document(
        "STANDALONE (₹ in crore)", (*_HEADER, _MERGED), "and again", (*_HEADER, _MERGED)
    )
    loaded: list[str] = []

    def load(name: str) -> Document:
        loaded.append(name)
        return document

    def build(add: int, tickers: list[str]) -> Path:
        return cli.table_page(
            "p",
            add,
            None,
            tickers,
            None,
            tmp_path / "lists",
            tmp_path / "out",
            load,
            lambda n: None,
            lambda p, n: "data:image/jpeg;base64,",
            manifest,
        )

    target = build(5, ["ITC"])
    kept = read_table_list(tmp_path / "lists" / "p.json")
    assert [c.document for c in kept.cards] == ["results/ITC-2024-06-30.pdf"] * 2, kept
    assert '"id": "t01"' in target.read_text(encoding="utf-8")
    loaded.clear()
    build(0, [])
    assert loaded == ["results/ITC-2024-06-30.pdf"], "a rebuild reads only the kept tables"
    assert read_table_list(tmp_path / "lists" / "p.json") == kept


def test_the_command_refuses_a_draw_with_no_documents(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no readable document"):
        cli.table_page(
            "p", 5, "SEC copy", [], None, tmp_path, tmp_path, manifest={"results/X.pdf": _ENTRY}
        )


def test_the_table_page_flags_reach_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "table_page", lambda *a: seen.append(a))
    cli.main(
        [
            "table-page",
            "--name",
            "p",
            "--add",
            "3",
            "--kind",
            "SEC copy",
            "--tickers",
            "ITC",
            "--url",
            "u",
        ]
    )
    assert seen == [("p", 3, "SEC copy", ["ITC"], "u")]


@pytest.mark.parametrize(
    ("pools", "says"),
    [
        pytest.param(("flagged", "any"), "half flagged by an audit", id="a drawn batch"),
        pytest.param(
            ("differs", "differs"), "most cross-checked facts differ", id="by differences"
        ),
    ],
)
def test_the_lede_says_how_the_tables_were_chosen(pools: tuple[str, ...], says: str) -> None:
    cards: list[dict[str, object]] = [{"id": f"t{n}", "pool": p} for n, p in enumerate(pools)]
    assert says in build_table_page(cards, _TEMPLATE, ("OK",))
