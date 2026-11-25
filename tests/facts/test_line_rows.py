"""`facts.line_rows` and the re-rowed loading of filed PDFs in `facts.documents`."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from docling_core.types.doc.document import DoclingDocument
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    RectangleObject,
)

from aigent.facts import documents, line_rows
from aigent.facts.line_rows import Layout, PdfBox, Word
from aigent.retrieval.chunk import Document
from aigent.retrieval.corpus import assembled, elements_of, table_rows

from ..retrieval.conftest import Grid, build_pdf, docling


def _chars(
    text: str, left: float, bottom: float = 100.0, width: float = 5.0
) -> list[line_rows.Char]:
    """Characters printed side by side from `left`, a space given no box."""
    return [
        (c, None if c == " " else (left + i * width, bottom, left + (i + 1) * width, bottom + 8))
        for i, c in enumerate(text)
    ]


@pytest.mark.parametrize(
    ("chars", "runs", "words", "spaced"),
    [
        pytest.param(
            _chars("Other income", 0),
            [("Other income", (0, 100, 60, 108))],
            ["Other", "income"],
            [True, False],
            id="a run splits at the spaces between its characters",
        ),
        pytest.param(
            _chars("143.74 174.56 204.80", 0),
            [("143.74 174.56 204.80", (0, 100, 100, 108))],
            ["143.74", "174.56", "204.80"],
            [True, True, False],
            id="a run of figures across columns splits into each figure",
        ),
        pytest.param(
            _chars("11", 0) + _chars(",", 10) + _chars("216.4", 15),
            [("11", (0, 100, 10, 108)), (",", (10, 100, 15, 108)), ("216.4", (15, 100, 40, 108))],
            ["11", ",", "216.4"],
            [False, False, False],
            id="runs with no space between them stay apart, each running on",
        ),
        pytest.param(
            [],
            [("Revenue ", (0, 100, 35, 108))],
            ["Revenue"],
            [True],
            id="a run none of whose characters is found is kept whole",
        ),
    ],
)
def test_split_runs(
    chars: list[line_rows.Char],
    runs: list[tuple[str, PdfBox]],
    words: list[str],
    spaced: list[bool],
) -> None:
    split = line_rows.split_runs(chars, runs)
    assert [(w, s) for w, _, s in split] == list(zip(words, spaced, strict=True)), split


def test_a_character_is_taken_by_one_run_only() -> None:
    runs: list[tuple[str, PdfBox]] = [("Revenue", (0, 100, 35, 108)), ("Revenue", (0, 98, 40, 110))]
    first, second = line_rows.split_runs(_chars("Revenue", 0), runs)
    assert second == ("Revenue", (0, 98, 40, 110), False), "the second run is kept whole"
    assert first == ("Revenue", (0, 100, 35, 108), False), first


def test_a_split_word_is_boxed_by_its_own_characters() -> None:
    split = line_rows.split_runs(_chars("12 345", 0), [("12 345", (0, 100, 30, 108))])
    assert split == [("12", (0, 100, 10, 108), True), ("345", (15, 100, 30, 108), False)], split


@pytest.mark.parametrize(
    ("rotation", "box"),
    [
        pytest.param(0, (10, 180, 30, 190), id="an upright page flips its height"),
        pytest.param(90, (10, 10, 20, 30), id="turned a quarter clockwise"),
        pytest.param(180, (70, 10, 90, 20), id="turned upside down"),
        pytest.param(270, (180, 70, 190, 90), id="turned three quarters clockwise"),
        pytest.param(-90, (180, 70, 190, 90), id="a negative turn counts the other way"),
    ],
)
def test_displayed(rotation: int, box: tuple[float, ...]) -> None:
    # a 20 by 10 box 10 from the left and bottom of a 100 by 200 page
    assert line_rows.displayed((10, 10, 30, 20), rotation, 100, 200) == box


_TABLE = Layout(
    box=(0, 0, 300, 200),
    header=1,
    body_top=50,
    columns={0: (0, 100), 1: (120, 180), 2: (200, 260)},
    width=3,
)
_HEADER = (("Particulars", "Jun", "Mar"),)


def _w(text: str, left: float, top: float, width: float = 30.0, spaced: bool = True) -> Word:
    return (text, (left, top, left + width, top + 8), spaced)


@pytest.mark.parametrize(
    ("words", "rows"),
    [
        pytest.param(
            [
                _w("Revenue", 0, 60),
                _w("100", 140, 60),
                _w("90", 220, 60),
                _w("Other", 0, 80),
                _w("income", 35, 80),
                _w("5", 150, 80),
                _w("4", 230, 80),
            ],
            (("Revenue", "100", "90"), ("Other income", "5", "4")),
            id="each printed line a row, its words in reading order",
        ),
        pytest.param(
            [_w("Revenue", 0, 60), _w("100", 140, 63), _w("90", 220, 58)],
            (("Revenue", "100", "90"),),
            id="words within half a line of each other share it",
        ),
        pytest.param(
            [_w("Revenue", 0, 60), _w("100", 140, 66)],
            (("Revenue", "", ""), ("", "100", "")),
            id="a word more than half a line down starts the next",
        ),
        pytest.param(
            [_w("Total", 0, 60), _w("17", 140, 60, 8), _w(",", 150, 60, 2), _w("009", 155, 60, 12)],
            (("Total", "17,009", ""),),
            id="a figure printed with spaced separators closes up",
        ),
        pytest.param(
            [_w("Chan", 0, 60, 20, spaced=False), _w("aes", 22, 60, 15)],
            (("Chanaes", "", ""),),
            id="a word with no space after it runs on into the next",
        ),
        pytest.param(
            [_w("Chan", 0, 60, 20, spaced=False), _w("aes", 40, 60, 15)],
            (("Chan aes", "", ""),),
            id="but not into one starting more than a line's height away",
        ),
        pytest.param(
            [_w("Revenue", 0, 60), _w("100", 105, 60, 10)],
            (("Revenue", "100", ""),),
            id="a word overlapping no column goes to the nearest",
        ),
        pytest.param(
            [_w("Quarter", 0, 30), _w("Revenue", 0, 60), _w("Note", 0, 230), _w("7", 320, 60)],
            (("Revenue", "", ""),),
            id="words above the body or outside the table are left out",
        ),
    ],
)
def test_line_rows(words: list[Word], rows: tuple[tuple[str, ...], ...]) -> None:
    assert line_rows.line_rows(_HEADER, _TABLE, words) == _HEADER + rows


def test_a_table_with_no_word_in_its_body_is_not_re_rowed() -> None:
    assert line_rows.line_rows(_HEADER, _TABLE, [_w("Quarter", 0, 30)]) is None


# A Docling table laid out between 600 and 700 points up the page, three rows of three columns
# each 100 wide from 50; its last row is two printed lines Docling merged.
_MERGED = Grid(
    rows=(
        ("Particulars", "Jun", "Mar"),
        ("Revenue", "100", "90"),
        ("Other income Total", "5 105", "4 94"),
    ),
    header_rows=1,
    box=(50, 600, 350, 700),
)
_PRINTED = [  # (left, baseline, text) as the page prints it
    (55, 680, "Particulars"),
    (160, 680, "Jun"),
    (260, 680, "Mar"),
    (55, 650, "Revenue"),
    (160, 650, "100"),
    (260, 650, "90"),
    (55, 630, "Other income"),
    (160, 630, "5"),
    (260, 630, "4"),
    (55, 610, "Total"),
    (160, 610, "105"),
    (260, 610, "94"),
]


def _printed(
    path: Path, lines: Sequence[tuple[float, float, str]], crop: tuple[float, float] = (0, 0)
) -> Path:
    """A one-page PDF printing each text at its left edge and baseline, its crop box starting
    `crop` from the media box's corner."""
    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    page.cropbox = RectangleObject((crop[0], crop[1], 612, 792))
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(
        "\n".join(f"BT /F1 6 Tf {x} {y} Td ({text}) Tj ET" for x, y, text in lines).encode()
    )
    page[NameObject("/Contents")] = ArrayObject([writer._add_object(stream)])
    with path.open("wb") as out:
        writer.write(out)
    return path


@pytest.mark.parametrize(
    ("crop", "left", "bottom"),
    [
        pytest.param((0, 0), 55, 628, id="a page shown whole"),
        pytest.param((10, 20), 45, 608, id="a cropped page measured from its crop box"),
    ],
)
def test_page_words_reads_each_word_and_where_it_sits(
    tmp_path: Path, crop: tuple[float, float], left: float, bottom: float
) -> None:
    pdf = pdfium.PdfDocument(_printed(tmp_path / "p.pdf", [(55, 630, "Other income 5")], crop))
    words = line_rows.page_words(pdf[0])
    assert [w for w, _, _ in words] == ["Other", "income", "5"], words
    assert all(
        left <= box[0] < box[2] and bottom <= box[1] < box[3] <= bottom + 12 for _, box, _ in words
    ), words


def _rendered(grid: Grid) -> tuple[Document, DoclingDocument]:
    parsed = docling([[grid]])
    return assembled("D", elements_of(parsed)), parsed


def test_a_table_docling_merged_is_re_rowed_by_its_printed_lines(tmp_path: Path) -> None:
    document, parsed = _rendered(_MERGED)
    words = documents.read_words(_printed(tmp_path / "p.pdf", _PRINTED), [1])
    [table] = documents.with_line_rows(document, parsed, words).tables
    assert table.rows == (
        ("Particulars", "Jun", "Mar"),
        ("Revenue", "100", "90"),
        ("Other income", "5", "4"),
        ("Total", "105", "94"),
    ), table.rows
    assert table.header_rows == 1


_BOX = (50, 600, 350, 700)


@pytest.mark.parametrize(
    ("grid", "printed", "split"),
    [
        pytest.param(
            Grid(rows=(("Particulars", "Note"), ("Revenue", "see below")), box=_BOX),
            [(55, 650, "Revenue")],
            False,
            id="no row of two figures",
        ),
        pytest.param(_MERGED, _PRINTED, True, id="a table split across two pages"),
        pytest.param(
            Grid(rows=(("", "", "", ""),), spans=((0, 0, 2, "100"), (0, 2, 4, "90")), box=_BOX),
            [(55, 650, "100"), (210, 650, "90")],
            False,
            id="no cell of a single column to place columns by",
        ),
        pytest.param(_MERGED, _PRINTED[:3], False, id="no printed word in its body"),
    ],
)
def test_a_table_that_cannot_be_re_rowed_stays_as_docling_read_it(
    tmp_path: Path, grid: Grid, printed: list[tuple[float, float, str]], split: bool
) -> None:
    document, parsed = _rendered(grid)
    if split:
        item = parsed.tables[0]
        item.prov.append(item.prov[0].model_copy())
    words = documents.read_words(_printed(tmp_path / "p.pdf", printed), [1])
    assert documents.with_line_rows(document, parsed, words).tables == document.tables


def test_a_re_rowed_table_keeps_its_caption_and_doclings_marks_over_the_rows_kept(
    tmp_path: Path,
) -> None:
    grid = Grid(rows=_MERGED.rows, header_rows=2, sections=(0, 2), box=_BOX, caption="Results")
    document, parsed = _rendered(grid)
    [before] = document.tables
    assert (before.header_rows, before.sections) == (3, (1, 3))
    words = documents.read_words(_printed(tmp_path / "p.pdf", _PRINTED), [1])
    [table] = documents.with_line_rows(document, parsed, words).tables
    assert table.rows[:2] == (("Results",), ("Particulars", "Jun", "Mar")), table.rows
    assert (table.header_rows, table.sections) == (2, (1,)), table


def test_a_rendered_table_matching_none_of_doclings_is_refused(tmp_path: Path) -> None:
    document, _ = _rendered(_MERGED)
    _, parsed = _rendered(Grid(rows=(("Cash", "1", "2"),), box=(50, 600, 350, 700)))
    words = documents.read_words(_printed(tmp_path / "p.pdf", _PRINTED), [1])
    with pytest.raises(ValueError, match="matches none of Docling's"):
        documents.with_line_rows(document, parsed, words)


def test_a_filed_pdf_is_re_rowed_once_and_a_re_rowing_change_reuses_its_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _printed(tmp_path / "results.pdf", _PRINTED)
    cache = tmp_path / "cache"
    _, parsed = _rendered(_MERGED)
    loads: list[str] = []
    reads: list[Sequence[int]] = []

    def parse(path: Path, doc_id: str, cache_dir: Path) -> DoclingDocument:
        loads.append("parsed")
        return parsed

    def read(path: Path, pages: Sequence[int]) -> dict[int, list[Word]]:
        reads.append(pages)
        return documents.read_words(path, pages)

    def words(
        path: Path, doc_id: str, reading: DoclingDocument, cache_dir: Path
    ) -> dict[int, list[Word]]:
        return documents.load_words_cached(path, doc_id, reading, cache_dir, read)

    first = documents.load_lines_cached(path, "D", cache, parse, words)
    again = documents.load_lines_cached(path, "D", cache, parse, words)
    assert (loads, reads) == (["parsed"], [[1]]), "the second load read the cache"
    assert again.tables == first.tables and len(first.tables[0].rows) == 4
    kept = documents.lines_file(path, "D", cache)
    kept.write_text("{not json", encoding="utf-8")
    documents.load_lines_cached(path, "D", cache, parse, words)
    assert (len(loads), len(reads)) == (2, 1), "a broken file is rebuilt from the kept words"
    monkeypatch.setattr(documents, "_LINES_FINGERPRINT", "0123456789ab")
    documents.load_lines_cached(path, "D", cache, parse, words)
    assert (len(loads), len(reads)) == (3, 1), "a re-rowing change reads the PDF no more"
    assert not kept.exists(), "the superseded file is swept"


def test_a_filed_pdfs_words_are_read_once_for_its_pages_with_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _printed(tmp_path / "results.pdf", _PRINTED)
    cache = tmp_path / "cache"
    parsed = docling([[], [_MERGED], []])
    reads: list[Sequence[int]] = []

    def read(path: Path, pages: Sequence[int]) -> dict[int, list[Word]]:
        reads.append(pages)
        return {page: [("100", (1.0, 2.0, 3.0, 4.0), False)] for page in pages}

    first = documents.load_words_cached(path, "D", parsed, cache, read)
    again = documents.load_words_cached(path, "D", parsed, cache, read)
    assert reads == [[2]] and again == first == {2: [("100", (1.0, 2.0, 3.0, 4.0), False)]}
    kept = documents.words_file(path, "D", cache)
    kept.write_bytes(b"not gzip")
    documents.load_words_cached(path, "D", parsed, cache, read)
    assert len(reads) == 2, "a file that does not read back is rebuilt"
    monkeypatch.setattr(documents, "_WORDS_FINGERPRINT", "0123456789ab")
    documents.load_words_cached(path, "D", parsed, cache, read)
    assert len(reads) == 3 and not kept.exists(), "the superseded file is swept"


def test_a_filed_pdf_is_read_re_rowed_by_default() -> None:
    default = inspect.signature(documents.load_verify_document).parameters["pdf"].default
    assert default is documents.load_lines_cached


def test_a_columns_edges_are_its_median_cells_so_a_stray_wide_cell_moves_none(
    tmp_path: Path,
) -> None:
    _, parsed = _rendered(_MERGED)
    item = parsed.tables[0]
    height = parsed.pages[1].size.height
    [stray] = [
        c
        for c in item.data.table_cells
        if (c.start_row_offset_idx, c.start_col_offset_idx) == (2, 1)
    ]
    assert stray.bbox is not None
    stray.bbox = stray.bbox.model_copy(update={"l": 60.0})
    shape = line_rows.layout(item, table_rows(item.data), height)
    assert shape is not None and shape.columns[1] == (150.0, 250.0), shape


def test_a_pdf_is_read_with_its_rupee_fonts_as_the_rupee_sign(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(build_pdf([["({`} crore)"]], second_font="Rupee"))
    pdf = documents._opened(path)
    words = "".join(w for w, _, _ in line_rows.page_words(pdf[0]))
    assert "₹" in words and "`" not in words, words


def test_a_pdfs_words_are_kept_to_a_hundredth_of_a_point(tmp_path: Path) -> None:
    [(page, words)] = documents.read_words(_printed(tmp_path / "p.pdf", _PRINTED[:3]), [1]).items()
    assert page == 1 and [w for w, _, _ in words] == ["Particulars", "Jun", "Mar"], words
    assert all(round(c, 2) == c for _, box, _ in words for c in box), words
