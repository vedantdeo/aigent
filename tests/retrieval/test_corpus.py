"""The corpus: a PDF in, a `Document` with page offsets, tables and headings out.

The load-bearing table is the one on page offsets. Every citation downstream names a page, and
offsets taken before normalisation are wrong in a way nothing else in the pipeline notices. PDFs
are read by a fake Docling (`conftest.fake_docling`), so no test downloads its models.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from docling.datamodel.base_models import DocumentStream
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from pypdf import PdfReader
from pypdf.generic import DictionaryObject, StreamObject

from aigent.config import PAGE_SEPARATOR
from aigent.retrieval import corpus
from aigent.retrieval.corpus import (
    FY25_REPORTS,
    assembled,
    cache_file,
    captions_restored,
    elements_of,
    load_corpus,
    load_pdf,
    load_pdf_cached,
    normalise,
    parse_cached,
    parse_file,
    read_pages,
    rupees_decoded,
    strip_furniture,
    table_rows,
    to_document,
)

from .conftest import Footer, Grid, Heading, Picture, Piece, Running, docling

pytestmark = pytest.mark.usefixtures("fake_docling")

HEADER = "RELIANCE INDUSTRIES LIMITED | INTEGRATED ANNUAL REPORT 2024-25"


def build_pdf(
    pages: list[list[str]],
    second_font: str = "Courier",
    encoding: bytes = b"/WinAnsiEncoding",
    to_unicode: dict[str, str] | None = None,
) -> bytes:
    """An uncompressed PDF with these lines on these pages, so a test can read one back. A part of
    a line in braces is drawn in `second_font`, with this encoding and, given one, a ToUnicode map
    from each character drawn to the text it reads as."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog, pages_obj = add(b""), add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    extra = b""
    if to_unicode is not None:
        pairs = "\n".join(f"<{ord(a):04X}> <{ord(b):04X}>" for a, b in to_unicode.items())
        cmap = (
            "begincmap\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
            f"{len(to_unicode)} beginbfchar\n{pairs}\nendbfchar\nendcmap\n"
        ).encode()
        mapped = add(
            b"<< /Length " + str(len(cmap)).encode() + b" >>\nstream\n" + cmap + b"\nendstream"
        )
        extra = b" /ToUnicode " + str(mapped).encode() + b" 0 R"
    font2 = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /"
        + second_font.encode()
        + b" /Encoding "
        + encoding
        + extra
        + b" >>"
    )

    kids: list[int] = []
    for lines in pages:
        text = b"BT /F1 12 Tf 72 720 Td 14 TL\n"
        for line in lines:
            for number, part in enumerate(re.split(r"[{}]", line)):
                escaped = part.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
                text += b"/F2 12 Tf " if number % 2 else b"/F1 12 Tf "
                text += b"(" + escaped.encode("cp1252") + b") Tj\n"
            text += b"T*\n"
        text += b"ET"
        content = add(
            b"<< /Length " + str(len(text)).encode() + b" >>\nstream\n" + text + b"\nendstream"
        )
        kids.append(
            add(
                b"<< /Type /Page /Parent " + str(pages_obj).encode() + b" 0 R "
                b"/MediaBox [0 0 612 792] /Contents " + str(content).encode() + b" 0 R "
                b"/Resources << /Font << /F1 "
                + str(font).encode()
                + b" 0 R /F2 "
                + str(font2).encode()
                + b" 0 R >> >> >>"
            )
        )

    objects[catalog - 1] = b"<< /Type /Catalog /Pages " + str(pages_obj).encode() + b" 0 R >>"
    objects[pages_obj - 1] = (
        b"<< /Type /Pages /Kids [" + b" ".join(f"{kid} 0 R".encode() for kid in kids) + b"] "
        b"/Count " + str(len(kids)).encode() + b" >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root "
        + str(catalog).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def annual_report(directory: Path, stem: str = "RELIANCE-FY25") -> Path:
    """Six pages of plausible report text behind a running header and a page number."""
    body = [
        ["Revenue from operations rose to Rs. 9,74,864 crore for the year."],
        ["The Board has recommended a final dividend of Rs. 10 per equity share."],
        ["Retail delivered an EBITDA of Rs. 25,053 crore, up 8.6 per cent."],
        ["Jio closed the year with 488 million subscribers across the network."],
        ["Capital expenditure for the year stood at Rs. 1,31,107 crore."],
        ["The Company employed 3,89,000 people as at 31 March 2025."],
    ]
    pages = [[HEADER, *lines, str(11 + number)] for number, lines in enumerate(body, start=1)]
    path = directory / f"{stem}.pdf"
    path.write_bytes(build_pdf(pages))
    return path


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        pytest.param("the ﬁnancial year", "the financial year", id="a ligature folds to ASCII"),
        pytest.param("Rs.\u00a09,74,864", "Rs. 9,74,864", id="a non-breaking space is a space"),
        pytest.param("Revenue\t\trose", "Revenue rose", id="a tab is a space"),
        pytest.param("Revenue    rose", "Revenue rose", id="a run of spaces collapses to one"),
        pytest.param("Segment   \nRevenue", "Segment\nRevenue", id="trailing spaces go"),
        pytest.param(
            "manage-\nment discussion", "management discussion", id="a break-hyphen rejoins"
        ),
        pytest.param("well-known brands", "well-known brands", id="a hyphen inside a line stays"),
        pytest.param("a\n\n\n\nb", "a\n\nb", id="a run of blank lines collapses to one"),
        pytest.param("page\x0cbreak", "page\nbreak", id="a form feed is a line break"),
        pytest.param("Revenue\x00 rose", "Revenue rose", id="any other control character goes"),
        pytest.param("  \n Revenue rose. \n  ", "Revenue rose.", id="the page itself is stripped"),
    ],
)
def test_normalise_cleans_extracted_text_without_moving_what_a_quote_matches(
    raw: str, cleaned: str
) -> None:
    assert normalise(raw) == cleaned


# Distinct sentences on purpose: lines differing only by a number share a digits-masked skeleton,
# so a fixture built from "Body 1", "Body 2" would be eaten by the very rule under test.
BODY = [
    ("Revenue from operations rose to Rs. 9,74,864 crore.", "Segment detail follows overleaf."),
    ("The Board recommended a final dividend of Rs. 10 per share.", "A record date will be set."),
    ("Retail delivered an EBITDA of Rs. 25,053 crore.", "Store count grew across formats."),
    ("Jio closed the year with 488 million subscribers.", "Churn stayed broadly flat."),
    ("Capital expenditure stood at Rs. 1,31,107 crore.", "Gearing remains conservative."),
]


def report_pages() -> list[str]:
    """Six pages carrying a running header, a page number, and a label repeated mid-page."""
    pages = [
        f"{HEADER} {11 + n}\n{opening}\n(Rs. in crore)\n{closing}\n{11 + n}"
        for n, (opening, closing) in enumerate(BODY)
    ]
    return [*pages, f"{HEADER} 16\n16"]


def test_strip_furniture_drops_what_repeats_at_the_edges_of_a_page() -> None:
    stripped = strip_furniture(report_pages(), ratio=0.5)

    assert len(stripped) == len(report_pages()), "dropping a page renumbers every citation after it"
    assert stripped[0] == f"{BODY[0][0]}\n(Rs. in crore)\n{BODY[0][1]}"
    assert not any(HEADER in page for page in stripped), (
        "the header carries the page number, so it never repeats verbatim - mask the digits"
    )
    assert all("(Rs. in crore)" in page for page in stripped[:5]), (
        "a label repeated mid-page is content, however often it appears"
    )
    assert stripped[5] == "", "a page that was nothing but furniture stays, empty"


def test_a_document_too_short_to_tell_keeps_everything() -> None:
    pages = [f"{HEADER}\nBody {n}" for n in range(1, 4)]

    assert strip_furniture(pages, ratio=0.5) == pages, (
        "on three pages a line on every page is as likely to be a real heading"
    )


def test_page_starts_index_the_joined_text_so_every_page_cites_itself() -> None:
    pages = ["First page.", "", "Third page is longer than the others.", "Fourth."]

    document = to_document("RIL-FY25", pages, source="corpus/RIL-FY25.pdf")

    assert document.page_starts[0] == 0, "page one starts where the text does"
    assert len(document.page_starts) == len(pages)
    assert document.text == PAGE_SEPARATOR.join(pages)
    for number, page in enumerate(pages, start=1):
        if not page:
            continue
        assert document.page_of(document.text.index(page)) == number, page
    assert document.page_of(len(document.text) - 1) == len(pages), (
        "the last character is on the last page"
    )


def test_a_pdf_arrives_as_one_document_with_its_furniture_gone(tmp_path: Path) -> None:
    path = annual_report(tmp_path)

    document = load_pdf(path)

    assert document.doc_id == "RELIANCE-FY25", "the filename stem names the document"
    assert document.source == str(path)
    assert HEADER not in document.text, "the running header repeats on all six pages"
    assert "\n12\n" not in document.text, "and so does a bare page number"
    assert "final dividend of Rs. 10 per equity share" in document.text
    assert document.page_of(document.text.index("488 million")) == 4


def test_the_corpus_is_every_pdf_in_filename_order(tmp_path: Path) -> None:
    annual_report(tmp_path, "TCS-FY25")
    annual_report(tmp_path, "INFY-FY25")
    (tmp_path / "notes.txt").write_text("not a report")
    (tmp_path / "training").mkdir()
    annual_report(tmp_path / "training", "HDFCBANK-FY25")

    documents = load_corpus(tmp_path)

    assert [document.doc_id for document in documents] == ["INFY-FY25", "TCS-FY25"], (
        "a training report in corpus/training/ must never be retrieved over"
    )


def test_a_document_set_loads_exactly_those_reports_in_filename_order(tmp_path: Path) -> None:
    for doc_id in ("TCS-FY24", "TCS-FY25", "INFY-FY24", "INFY-FY25"):
        annual_report(tmp_path, doc_id)

    documents = load_corpus(tmp_path, doc_ids=["TCS-FY25", "INFY-FY25"])

    assert [document.doc_id for document in documents] == ["INFY-FY25", "TCS-FY25"]


def test_a_document_set_naming_a_missing_report_raises(tmp_path: Path) -> None:
    annual_report(tmp_path, "TCS-FY25")

    with pytest.raises(ValueError, match="INFY-FY25"):
        load_corpus(tmp_path, doc_ids=["TCS-FY25", "INFY-FY25"])


def test_the_published_evals_read_the_fy25_reports() -> None:
    assert FY25_REPORTS == ("ITC-FY25", "RELIANCE-FY25", "TATAMOTORS-FY25")


def test_a_corpus_with_no_pdfs_raises_rather_than_scoring_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no PDFs"):
        load_corpus(tmp_path)


def test_read_pages_returns_one_string_per_page(tmp_path: Path) -> None:
    pages = read_pages(annual_report(tmp_path))

    assert len(pages) == 6
    assert "Revenue from operations" in pages[0]
    assert "3,89,000 people" in pages[5]


# --- Docling's elements --------------------------------------------------------------------------

SCALE = "(Rs in crore)"
TABLE = Grid(
    rows=((SCALE, "", ""), ("Particulars", "2025", "2024"), ("Revenue", "9,74,864", "9,01,064")),
    spans=((0, 0, 3, SCALE),),
)


def test_a_page_s_elements_keep_its_reading_order_and_kinds() -> None:
    document = docling(
        [
            [
                Heading("Statement of Profit and Loss"),
                TABLE,
                "The notes are part of it.",
                Footer("12"),
            ],
            [],
            ["Revenue rose by 8 per cent."],
        ]
    )

    first, empty, last = elements_of(document)

    assert [(e.heading, e.is_table) for e in first] == [
        (True, False),
        (False, True),
        (False, False),
    ]
    assert first[1].text == f"{SCALE}\nParticulars 2025 2024\nRevenue 9,74,864 9,01,064"
    assert empty == [], "a page with nothing on it keeps its place"
    assert [e.text for e in last] == ["Revenue rose by 8 per cent."]
    assert all("12" != e.text for e in first), "a page footer Docling set aside stays out"


def test_text_drawn_inside_a_picture_is_read_where_the_picture_stands() -> None:
    document = docling(
        [["Dividend", Picture(("6.50 per share", "(Interim)")), "declared for the year."]]
    )

    [page] = elements_of(document)

    assert [e.text for e in page] == [
        "Dividend",
        "6.50 per share",
        "(Interim)",
        "declared for the year.",
    ]


def test_a_cell_spanning_columns_is_given_once_in_its_first() -> None:
    document = docling([[TABLE]])
    [table] = document.tables

    rows = table_rows(table.data)

    assert rows[0] == (SCALE, "", "") and rows[2] == ("Revenue", "9,74,864", "9,01,064")


@pytest.mark.parametrize(("marked", "counted"), [(0, 0), (1, 1), (2, 2)])
def test_a_table_counts_the_leading_rows_docling_marked_as_column_headers(
    marked: int, counted: int
) -> None:
    grid = Grid(rows=TABLE.rows, spans=TABLE.spans, header_rows=marked)

    [[element]] = elements_of(docling([[grid]]))

    assert element.header_rows == counted


def test_furniture_goes_from_text_but_never_from_a_heading_or_a_table() -> None:
    """The scale line atop every statement page repeats like a running header; it must stay."""
    page = [Heading("Balance Sheet"), TABLE, "Notes follow.", "NOTES"]
    elements = elements_of(docling([page] * 5))

    document = assembled("BAJAJ-AUTO-FY24", elements)

    assert document.text.count(SCALE) == 5, "the scale line is in every table still"
    assert document.text.count("Balance Sheet") == 5, "and every heading"
    assert "NOTES" not in document.text, "the section tab that repeats on every page is gone"


def test_each_table_and_heading_spans_exactly_its_text_on_its_page() -> None:
    pages = [["Opening remarks."], [Heading("Statement of Profit and Loss"), TABLE, "Notes."]]

    document = assembled("RELIANCE-FY25", elements_of(docling(pages)))

    [table] = document.tables
    [heading] = document.headings
    assert document.text[heading.start : heading.end] == "Statement of Profit and Loss"
    assert document.text[table.start : table.end].startswith(SCALE)
    assert document.text[table.start : table.end].endswith("9,01,064")
    assert table.rows[2] == ("Revenue", "9,74,864", "9,01,064")
    assert document.page_of(table.start) == document.page_of(heading.start) == 2


def test_tables_and_headings_survive_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = replace(TABLE, sections=(2,))
    pages = [
        [Running("STANDALONE NOTES"), Heading("Statement of Profit and Loss"), table, "Notes."]
    ]
    monkeypatch.setattr(corpus, "convert", lambda path: docling(pages))
    path = annual_report(tmp_path)

    first = load_pdf_cached(path, cache_dir=tmp_path / ".cache")
    second = load_pdf_cached(path, cache_dir=tmp_path / ".cache")

    assert second == first and second.tables and second.headings
    assert second.tables[0].sections == (2,) and second.running == ("STANDALONE NOTES",)


# --- The document cache -----------------------------------------------------------------------


def _revised_report() -> bytes:
    """The same report reissued, with distinct lines per page so none of it reads as furniture."""
    body = [
        "Revenue from operations was restated to Rs. 9,00,000 crore.",
        "The dividend recommendation was withdrawn pending review.",
        "Retail EBITDA was corrected in the reissued statement.",
        "Subscriber numbers were re-presented on a comparable basis.",
        "Capital expenditure has been reclassified between segments.",
        "Headcount as at the year end was restated downwards.",
    ]
    return build_pdf([[HEADER, line, str(11 + n)] for n, line in enumerate(body, start=1)])


def test_a_second_load_is_served_from_cache_without_parsing_the_pdf_again(
    tmp_path: Path, fake_docling: list[Path]
) -> None:
    path = annual_report(tmp_path)

    first = load_pdf_cached(path, cache_dir=tmp_path / ".cache")
    second = load_pdf_cached(path, cache_dir=tmp_path / ".cache")

    assert len(fake_docling) == 1, "the second load re-parsed the PDF"
    assert second == first, "a cached document must be indistinguishable from a parsed one"
    assert second.page_starts == first.page_starts, "page offsets have to survive the round trip"


def test_republishing_the_report_invalidates_its_cache(
    tmp_path: Path, fake_docling: list[Path]
) -> None:
    """A company quietly reissuing a report is the case that would poison a filename-keyed cache."""
    path = annual_report(tmp_path)
    load_pdf_cached(path, cache_dir=tmp_path / ".cache")

    path.write_bytes(_revised_report())
    reloaded = load_pdf_cached(path, cache_dir=tmp_path / ".cache")

    assert len(fake_docling) == 2, "changed bytes must not be served from the old cache"
    assert "restated" in reloaded.text


def test_changing_the_ingestion_code_invalidates_the_cache(tmp_path: Path) -> None:
    """The half a content hash cannot see: same PDF, different `normalise`, different text."""
    path = annual_report(tmp_path)

    before = cache_file(path, "RELIANCE-FY25", tmp_path / ".cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(corpus, "_INGEST_FINGERPRINT", "0123456789ab")
        after = cache_file(path, "RELIANCE-FY25", tmp_path / ".cache")

    assert before != after, "editing the ingestion code must not reuse text it did not produce"


def test_a_truncated_cache_file_is_reparsed_rather_than_raising(tmp_path: Path) -> None:
    path = annual_report(tmp_path)
    cache = tmp_path / ".cache"
    load_pdf_cached(path, cache_dir=cache)
    written = cache_file(path, "RELIANCE-FY25", cache)
    written.write_text('{"doc_id": "RELIANCE-FY25", "text": "tru', encoding="utf-8")

    recovered = load_pdf_cached(path, cache_dir=cache)

    assert "final dividend of Rs. 10 per equity share" in recovered.text


def test_the_corpus_does_not_cache_unless_asked(tmp_path: Path) -> None:
    """Opt-in on purpose: reading the PDFs is always right, and a stale hit is wrong quietly."""
    annual_report(tmp_path)

    load_corpus(tmp_path)

    assert not (tmp_path / ".cache").exists(), "caching must be asked for, not assumed"


def test_the_cache_keeps_one_file_per_document(tmp_path: Path) -> None:
    path = annual_report(tmp_path)
    cache = tmp_path / ".cache"

    load_pdf_cached(path, cache_dir=cache)
    path.write_bytes(_revised_report())
    load_pdf_cached(path, cache_dir=cache)

    assert len(list(cache.glob("RELIANCE-FY25-*.json"))) == 1, "superseded entries should be swept"


def test_a_change_to_rendering_reads_the_kept_parse_rather_than_the_pdf(
    tmp_path: Path, fake_docling: list[Path]
) -> None:
    path = annual_report(tmp_path)
    cache = tmp_path / ".cache"
    first = load_pdf_cached(path, cache_dir=cache)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(corpus, "_INGEST_FINGERPRINT", "0123456789ab")
        second = load_pdf_cached(path, cache_dir=cache)

    assert len(fake_docling) == 1, "a rendering change re-parsed the PDF"
    assert second == first
    assert len(list(cache.glob("RELIANCE-FY25-*.json"))) == 1, "the old rendering is swept"
    assert parse_file(path, "RELIANCE-FY25", cache).exists(), "the sweep kept the parse"


def test_a_change_to_parsing_parses_the_pdf_again(tmp_path: Path, fake_docling: list[Path]) -> None:
    path = annual_report(tmp_path)
    cache = tmp_path / ".cache"
    load_pdf_cached(path, cache_dir=cache)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(corpus, "_PARSE_FINGERPRINT", "0123456789ab")
        patch.setattr(corpus, "_INGEST_FINGERPRINT", "ba9876543210")
        load_pdf_cached(path, cache_dir=cache)

    assert len(fake_docling) == 2, "a parse made by other parsing code was reused"
    assert len(list(cache.glob("RELIANCE-FY25.*.docling"))) == 1, "the old parse is swept"


def test_a_truncated_parse_is_parsed_again_rather_than_raising(
    tmp_path: Path, fake_docling: list[Path]
) -> None:
    path = annual_report(tmp_path)
    cache = tmp_path / ".cache"
    parse_cached(path, "RELIANCE-FY25", cache)
    parse_file(path, "RELIANCE-FY25", cache).write_text('{"schema_name": "Docl', encoding="utf-8")

    parsed = parse_cached(path, "RELIANCE-FY25", cache)

    assert len(fake_docling) == 2, fake_docling
    assert "final dividend" in parsed.export_to_text()


def test_republishing_the_report_keeps_one_parse(tmp_path: Path, fake_docling: list[Path]) -> None:
    path = annual_report(tmp_path)
    cache = tmp_path / ".cache"
    load_pdf_cached(path, cache_dir=cache)
    path.write_bytes(_revised_report())
    reloaded = load_pdf_cached(path, cache_dir=cache)

    assert len(fake_docling) == 2, "changed bytes must not be served from the old parse"
    assert "restated" in reloaded.text
    assert len(list(cache.glob("RELIANCE-FY25.*.docling"))) == 1, "superseded parses are swept"


DIFFERENCES = b"<< /BaseEncoding /WinAnsiEncoding /Differences [31 /grave] >>"


@pytest.mark.parametrize(
    ("line", "font", "encoding", "to_unicode", "read_as"),
    [
        pytest.param(
            "({`} crore)",
            "ABCDEF+RupeeForadian",
            b"/WinAnsiEncoding",
            None,
            "(₹ crore)",
            id="Foradian's backtick is the rupee",
        ),
        pytest.param(
            "{`}1,068.03 crore",
            "Rupee",
            b"/WinAnsiEncoding",
            None,
            "₹1,068.03 crore",
            id="Rupee's backtick is the rupee",
        ),
        pytest.param(
            "{`}205.36",
            "IndianRupee-Regular",
            b"/WinAnsiEncoding",
            None,
            "₹205.36",
            id="a styled family name is still its family",
        ),
        pytest.param(
            "({C} in millions)",
            "ITFRupee",
            b"/WinAnsiEncoding",
            None,
            "(₹ in millions)",
            id="ITFRupee's C is the rupee",
        ),
        pytest.param(
            "{H} 44,685 {J}9,723 {K}26 ({I} Cr) {D}10 {E}5 {L}9",
            "ITFRupee",
            b"/WinAnsiEncoding",
            None,
            "₹ 44,685 ₹9,723 ₹26 (₹ Cr) ₹10 ₹5 ₹9",
            id="every ITFRupee key seen in the reports is the rupee",
        ),
        pytest.param(
            "{\x1f}10",
            "Rupee",
            DIFFERENCES,
            None,
            "₹10",
            id="a backtick moved by Differences is still the rupee",
        ),
        pytest.param(
            "{` - ^}",
            "Rupee",
            b"/WinAnsiEncoding",
            {"`": "`", "-": "-", "^": "^"},
            "₹ - ^",
            id="a map already present changes only its backtick",
        ),
        pytest.param(
            "{R} 23,768",
            "RupeeForadian",
            b"/WinAnsiEncoding",
            None,
            "R 23,768",
            id="Foradian's R is printed as an R",
        ),
        pytest.param(
            "{Others China 118.6 – -}",
            "RupeeForadian",
            b"/WinAnsiEncoding",
            None,
            "Others China 118.6 – -",
            id="Foradian's letters, digits and dashes stay",
        ),
        pytest.param("{C}", "Rupee", b"/WinAnsiEncoding", None, "C", id="Rupee's C stays a C"),
        pytest.param(
            "{` A Z c 0}",
            "ITFRupee",
            b"/WinAnsiEncoding",
            None,
            "` A Z c 0",
            id="ITFRupee's other keys stay",
        ),
        pytest.param(
            "({`} crore) Class C",
            "Courier",
            b"/WinAnsiEncoding",
            None,
            "(` crore) Class C",
            id="a font that is not a rupee font is untouched",
        ),
        pytest.param(
            "{`}",
            "SuperRupee",
            b"/WinAnsiEncoding",
            None,
            "`",
            id="a family merely containing Rupee is untouched",
        ),
        pytest.param(
            "{`}",
            "Rupee",
            b"/MacRomanEncoding",
            None,
            "`",
            id="an encoding other than WinAnsi is untouched",
        ),
    ],
)
def test_only_a_rupee_font_s_rupee_keys_read_as_the_rupee_sign(
    tmp_path: Path,
    line: str,
    font: str,
    encoding: bytes,
    to_unicode: dict[str, str] | None,
    read_as: str,
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(build_pdf([[line]], font, encoding, to_unicode))
    source = rupees_decoded(path)
    reader = PdfReader(source.stream if isinstance(source, DocumentStream) else source)
    assert reader.pages[0].extract_text().strip() == read_as


def test_a_pdf_without_a_rupee_font_is_passed_on_untouched(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(build_pdf([["({`} crore)"]], second_font="Courier"))
    assert rupees_decoded(path) == path


def test_a_decoded_pdf_keeps_its_file_name_for_the_parser(tmp_path: Path) -> None:
    path = tmp_path / "HDFCBANK-FY25.pdf"
    path.write_bytes(build_pdf([["({`} crore)"]], second_font="Rupee"))
    source = rupees_decoded(path)
    assert isinstance(source, DocumentStream) and source.name == path.name, source


def test_a_code_moved_by_differences_is_mapped_by_its_own_code(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(build_pdf([["{\x1f}10"]], "Rupee", DIFFERENCES))
    source = rupees_decoded(path)
    assert isinstance(source, DocumentStream), source
    resources = cast(DictionaryObject, PdfReader(source.stream).pages[0]["/Resources"])
    font = cast(DictionaryObject, cast(DictionaryObject, resources["/Font"])["/F2"].get_object())
    cmap = cast(StreamObject, font["/ToUnicode"].get_object()).get_data()
    assert b"<1F> <20B9>" in cmap and b"<60> <20B9>" in cmap, cmap


# Cells tile x 50-490 in two columns, three rows from y 660 down; the caption band is y 660-700,
# the empty margin x 490-550, and the first row y 573-660.
LAID_OUT = Grid(
    rows=(("Particulars", "2025"), ("Revenue", "9,74,864"), ("Profit", "79,020")),
    box=(50, 400, 550, 700),
    band=40,
    margin=60,
)


def printed(text: str, left: float, bottom: float, right: float, top: float) -> corpus.Line:
    return text, BoundingBox(l=left, b=bottom, r=right, t=top, coord_origin=CoordOrigin.BOTTOMLEFT)


@pytest.mark.parametrize(
    ("before", "table", "lines", "restored"),
    [
        pytest.param(
            [],
            LAID_OUT,
            [printed("(₹ crore)", 430, 670, 480, 680)],
            ["(₹ crore)"],
            id="a scale line in the band above the first row is restored",
        ),
        pytest.param(
            [],
            LAID_OUT,
            [printed("(₹ crore)", 430, 665, 480, 675), printed("Gross block", 60, 685, 250, 695)],
            ["Gross block", "(₹ crore)"],
            id="several dropped lines come back top to bottom",
        ),
        pytest.param(
            [],
            LAID_OUT,
            [printed("(₹ In Crore)", 495, 600, 545, 610)],
            ["(₹ In Crore)"],
            id="a line level with the first row, beside its cells",
        ),
        pytest.param(
            [],
            LAID_OUT,
            [printed("Overview", 495, 450, 545, 460)],
            [],
            id="text beside the lower rows is not a caption",
        ),
        pytest.param(
            [],
            LAID_OUT,
            [printed("Revenue", 60, 520, 120, 530)],
            [],
            id="a line inside a cell is the cell's",
        ),
        pytest.param(
            [],
            LAID_OUT,
            [printed("Page 12", 560, 670, 600, 680)],
            [],
            id="a line outside the table's box is not its caption",
        ),
        pytest.param(
            ["Gross block"],
            LAID_OUT,
            [printed("Gross block", 60, 685, 250, 695)],
            ["Gross block"],
            id="a line printed elsewhere on the page is still this table's",
        ),
        pytest.param([], LAID_OUT, [], [], id="no lines read, nothing restored"),
        pytest.param(
            [],
            TABLE,
            [printed("(₹ crore)", 430, 670, 480, 680)],
            [],
            id="a table without cell boxes is left as it is",
        ),
    ],
)
def test_lines_dropped_from_a_table_s_box_become_its_leading_header_rows(
    before: list[str], table: Grid, lines: list[corpus.Line], restored: list[str]
) -> None:
    document = captions_restored(docling([[*before, table]]), {1: lines})

    [page] = elements_of(document)

    assert [e.text for e in page[:-1]] == before, "a restored line is the table's, not loose text"
    assert page[-1].rows is not None
    assert page[-1].rows[: len(restored)] == tuple((line,) for line in restored), page[-1].rows
    assert page[-1].header_rows == len(restored) + table.header_rows


def test_a_caption_docling_found_is_its_table_s_first_row_and_not_restored_twice() -> None:
    table = replace(LAID_OUT, caption="(₹ crore)")
    document = captions_restored(
        docling([[table]]), {1: [printed("(₹ crore)", 430, 670, 480, 680)]}
    )

    [page] = elements_of(document)

    assert len(page) == 1, "the caption is not printed again as text"
    assert page[0].rows is not None and page[0].rows[:2] == (
        ("(₹ crore)",),
        ("Particulars", "2025"),
    )
    assert page[0].header_rows == 1


def test_a_page_s_running_headers_are_its_context_and_never_its_text() -> None:
    notes = "Notes to standalone financial statements (Contd.)"
    body = [
        "Revenue rose.",
        "Margins held.",
        "Exports doubled.",
        "Debt was repaid.",
        "Cash grew.",
        "Stock fell.",
        "Tax was paid.",
        "Plants ran.",
        "Wages rose.",
    ]
    pages: list[list[Piece]] = [
        [Running("STANDALONE FINANCIAL STATEMENTS"), notes, line] for line in body
    ]
    pages.append(["The Board met six times."])

    document = assembled("BAJAJ-AUTO-FY24", elements_of(docling(pages)))

    assert document.running[0] == f"STANDALONE FINANCIAL STATEMENTS · {notes}", document.running
    assert document.running[9] == "", "a page without a running header has no context"
    assert "STANDALONE FINANCIAL STATEMENTS" not in document.text
    assert notes not in document.text, "a running line stripped as furniture moves to the context"
    assert "Revenue rose." in document.text


def test_a_table_carries_its_footnote_last_and_its_section_rows_past_its_caption() -> None:
    table = replace(TABLE, caption="(₹ crore)", footnote="* Less than ₹ 50 lakh", sections=(1,))

    [page] = elements_of(docling([[table]]))

    [element] = page
    assert element.rows is not None
    assert element.rows[0] == ("(₹ crore)",) and element.rows[-1] == ("* Less than ₹ 50 lakh",)
    assert element.header_rows == 1 + TABLE.header_rows, "a footnote is no header"
    assert element.sections == (2,), "a section row counts the caption above it"
