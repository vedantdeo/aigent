"""`facts.documents`: each filed document's format, text-less and OCR'd pages, read by format."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument
from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject

from aigent.facts import documents
from aigent.facts.cfg_facts import FACTS_TICKERS, VERIFY_TICKERS
from aigent.retrieval.chunk import Document

from ..retrieval.conftest import docling

_TEXT = "Interest earned 4858681 Other income 922986 Total income 5781667 " * 5
_HTML = """<html><body>
<h2>FINANCIAL RESULTS FOR THE QUARTER ENDED JUNE 30, 2023</h2>
<p>(&#8377; in lac)</p>
<table>
<tr><td>Particulars</td><td>30.06.2023</td><td>31.03.2023</td></tr>
<tr><td>Interest Earned</td><td>4858681</td><td>4511936</td></tr>
<tr><td>Other income</td><td>922986</td><td>873118</td></tr>
</table>
<p>Notes follow the table.</p>
</body></html>"""


def _pdf(path: Path, pages: list[str], modes: dict[int, list[int]] | None = None) -> Path:
    """A PDF with one page per string; an empty string is a page with no text layer. `modes` gives
    a 1-based page one text object per render mode (3 draws it invisible) instead of one visible."""
    writer = PdfWriter()
    for number, text in enumerate(pages, 1):
        page = writer.add_blank_page(612, 792)
        if not text:
            continue
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
        drawn = [
            f"BT /F1 6 Tf {mode} Tr 20 {700 - 20 * i} Td ({text}) Tj ET"
            for i, mode in enumerate((modes or {}).get(number, [0]))
        ]
        stream.set_data("\n".join(drawn).encode())
        page[NameObject("/Contents")] = ArrayObject([writer._add_object(stream)])
    with path.open("wb") as out:
        writer.write(out)
    return path


@pytest.fixture
def verify_dir(tmp_path: Path) -> Path:
    """A verify folder: a text PDF, a scanned PDF, a report with one photo page, a scan with OCR
    text behind, and HTML."""
    root = tmp_path / "verify"
    (root / "results").mkdir(parents=True)
    _pdf(root / "results/TEXT-2025-03-31.pdf", [_TEXT, _TEXT])
    _pdf(root / "results/SCAN-2025-03-31.pdf", [_TEXT, "", ""])
    _pdf(root / "results/PHOTO-FY25.pdf", [""] + [_TEXT] * 10)
    _pdf(root / "results/OCR-2025-03-31.pdf", [_TEXT] * 3, {2: [3], 3: [3]})
    (root / "results/WEB-2023-06-30.htm").write_text(_HTML, encoding="utf-8")
    names = (
        "TEXT-2025-03-31.pdf",
        "SCAN-2025-03-31.pdf",
        "PHOTO-FY25.pdf",
        "OCR-2025-03-31.pdf",
        "WEB-2023-06-30.htm",
    )
    manifest = {f"results/{n}": {"url": f"https://x/{n}", "sha256": "0"} for n in names}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        pytest.param([_TEXT, _TEXT], [], id="every page has text"),
        pytest.param([_TEXT, "", ""], [2, 3], id="a cover letter then scans"),
        pytest.param(["", _TEXT], [1], id="a scanned first page"),
        pytest.param(["short"], [1], id="a page under the floor counts as text-less"),
    ],
)
def test_text_less_pages(tmp_path: Path, pages: list[str], expected: list[int]) -> None:
    count, found = documents.text_less_pages(_pdf(tmp_path / "a.pdf", pages))
    assert count == len(pages)
    assert found == expected, found


@pytest.mark.parametrize(
    ("pages", "modes", "expected"),
    [
        pytest.param([_TEXT, _TEXT], {}, [], id="typed pages"),
        pytest.param([_TEXT] * 3, {2: [3], 3: [3]}, [2, 3], id="a cover letter then OCR'd scans"),
        pytest.param([_TEXT], {1: [3, 3, 0]}, [1], id="mostly invisible text"),
        pytest.param([_TEXT], {1: [0, 0, 3]}, [], id="mostly visible text"),
        pytest.param([_TEXT], {1: [0, 3]}, [], id="exactly half invisible"),
        pytest.param(["", _TEXT], {}, [], id="a page with no text at all"),
    ],
)
def test_ocr_pages(
    tmp_path: Path, pages: list[str], modes: dict[int, list[int]], expected: list[int]
) -> None:
    found = documents.ocr_pages(_pdf(tmp_path / "a.pdf", pages, modes))
    assert found == expected, found


@pytest.mark.parametrize(
    ("pages", "text_less", "ocr", "scan"),
    [
        pytest.param(10, [], [], False, id="no text-less page"),
        pytest.param(10, [1], [], False, id="exactly the share allowed"),
        pytest.param(10, [1, 2], [], True, id="over the share"),
        pytest.param(585, list(range(16)), [], False, id="an annual report's photo pages"),
        pytest.param(13, list(range(2, 14)), [], True, id="a scanned results filing"),
        pytest.param(24, [], list(range(3, 16)), True, id="a scan with OCR text behind"),
        pytest.param(10, [1], [2], True, id="text-less and OCR'd pages add up"),
        pytest.param(10, [1], [1], False, id="a page both text-less and OCR'd counts once"),
        pytest.param(0, [], [], False, id="html has no pages"),
    ],
)
def test_is_scan(pages: int, text_less: list[int], ocr: list[int], scan: bool) -> None:
    entry: documents.Entry = {"pages": pages, "text_less_pages": text_less, "ocr_pages": ocr}
    assert documents.is_scan(entry) is scan, entry


@pytest.mark.parametrize(
    ("name", "kind", "count", "pages", "ocr"),
    [
        pytest.param("results/TEXT-2025-03-31.pdf", "pdf", 2, [], [], id="a text pdf"),
        pytest.param("results/SCAN-2025-03-31.pdf", "pdf", 3, [2, 3], [], id="a scanned pdf"),
        pytest.param("results/PHOTO-FY25.pdf", "pdf", 11, [1], [], id="a report with a photo page"),
        pytest.param("results/OCR-2025-03-31.pdf", "pdf", 3, [], [2, 3], id="a scan with OCR text"),
        pytest.param("results/WEB-2023-06-30.htm", "html", 0, [], [], id="an html filing"),
    ],
)
def test_flag_documents_records_each_entry(
    verify_dir: Path, name: str, kind: str, count: int, pages: list[int], ocr: list[int]
) -> None:
    documents.flag_documents(verify_dir)
    entry = documents.read_manifest(verify_dir)[name]
    assert entry["format"] == kind, entry
    assert entry["pages"] == count, entry
    assert entry["text_less_pages"] == pages, entry
    assert entry["ocr_pages"] == ocr, entry
    assert entry["url"] == f"https://x/{name.removeprefix('results/')}", entry


def test_flag_documents_returns_the_scans(verify_dir: Path) -> None:
    scans = ["results/OCR-2025-03-31.pdf", "results/SCAN-2025-03-31.pdf"]
    assert documents.flag_documents(verify_dir) == scans


def test_flag_documents_refuses_an_unknown_format(verify_dir: Path) -> None:
    (verify_dir / "results/X.docx").write_bytes(b"")
    manifest = documents.read_manifest(verify_dir)
    manifest["results/X.docx"] = {"url": "https://x/X.docx"}
    documents.write_manifest(manifest, verify_dir)
    with pytest.raises(ValueError, match="X.docx"):
        documents.flag_documents(verify_dir)


def test_the_manifest_is_written_in_canonical_order(verify_dir: Path) -> None:
    documents.flag_documents(verify_dir)
    manifest = json.loads((verify_dir / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest) == sorted(manifest)
    for name, entry in manifest.items():
        assert list(entry) == sorted(entry), name


def test_an_html_filing_reads_as_one_page_with_its_table(verify_dir: Path) -> None:
    document = documents.load_html(verify_dir / "results/WEB-2023-06-30.htm", "WEB-2023-06-30")
    assert document.doc_id == "WEB-2023-06-30"
    assert len(document.tables) == 1
    assert document.tables[0].rows[1] == ("Interest Earned", "4858681", "4511936")
    assert "FINANCIAL RESULTS FOR THE QUARTER" in document.text
    assert "₹ in lac" in document.text
    assert document.text.index("Interest Earned") < document.text.index("Notes follow")


def _pdf_reader(path: Path, doc_id: str) -> Document:
    return Document(doc_id=f"pdf:{doc_id}", text="", source=str(path))


def _html_reader(path: Path, doc_id: str) -> Document:
    return Document(doc_id=f"html:{doc_id}", text="", source=str(path))


@pytest.mark.parametrize(
    ("name", "ocr_tickers", "doc_id"),
    [
        pytest.param(
            "results/TEXT-2025-03-31.pdf", (), "pdf:verify-results-TEXT-2025-03-31", id="a text pdf"
        ),
        pytest.param(
            "results/PHOTO-FY25.pdf",
            (),
            "pdf:verify-results-PHOTO-FY25",
            id="a few photo pages pass",
        ),
        pytest.param(
            "results/WEB-2023-06-30.htm",
            (),
            "html:verify-results-WEB-2023-06-30",
            id="an html filing",
        ),
        pytest.param(
            "results/OCR-2025-03-31.pdf",
            ("OCR",),
            "pdf:verify-results-OCR-2025-03-31",
            id="a scan with OCR text, of a company read so",
        ),
    ],
)
def test_load_verify_document_reads_by_format(
    verify_dir: Path, name: str, ocr_tickers: tuple[str, ...], doc_id: str
) -> None:
    documents.flag_documents(verify_dir)
    read = documents.load_verify_document(name, verify_dir, _pdf_reader, _html_reader, ocr_tickers)
    assert read.doc_id == doc_id, read.doc_id


@pytest.mark.parametrize(
    ("flag_first", "name", "ocr_tickers", "error"),
    [
        pytest.param(True, "results/NOPE.pdf", (), KeyError, id="a path not in the manifest"),
        pytest.param(
            False, "results/TEXT-2025-03-31.pdf", (), ValueError, id="an entry not flagged"
        ),
        pytest.param(True, "results/SCAN-2025-03-31.pdf", (), ValueError, id="a scanned pdf"),
        pytest.param(
            True,
            "results/SCAN-2025-03-31.pdf",
            ("SCAN",),
            ValueError,
            id="a scan with no text, even of a company read through OCR",
        ),
        pytest.param(
            True,
            "results/OCR-2025-03-31.pdf",
            ("TEXT",),
            ValueError,
            id="a scan with OCR text, of a company not read so",
        ),
    ],
)
def test_load_verify_document_refuses(
    verify_dir: Path,
    flag_first: bool,
    name: str,
    ocr_tickers: tuple[str, ...],
    error: type[Exception],
) -> None:
    if flag_first:
        documents.flag_documents(verify_dir)
    with pytest.raises(error):
        documents.load_verify_document(name, verify_dir, _pdf_reader, _html_reader, ocr_tickers)


@pytest.mark.parametrize(
    ("pages", "text_less", "ocr", "ocr_scan"),
    [
        pytest.param(10, [], [], False, id="typed pages"),
        pytest.param(24, [], list(range(3, 16)), True, id="a scan with OCR text behind"),
        pytest.param(10, [1], list(range(2, 11)), True, id="one text-less page among OCR'd"),
        pytest.param(10, [1, 2], list(range(3, 11)), False, id="text-less pages over the share"),
        pytest.param(13, list(range(2, 14)), [], False, id="a scan with no text"),
    ],
)
def test_is_ocr_scan(pages: int, text_less: list[int], ocr: list[int], ocr_scan: bool) -> None:
    entry: documents.Entry = {"pages": pages, "text_less_pages": text_less, "ocr_pages": ocr}
    assert documents.is_ocr_scan(entry) is ocr_scan, entry


@pytest.mark.parametrize(
    ("name", "ticker"),
    [
        ("results/BAJAJ-AUTO-2025-09-30.pdf", "BAJAJ-AUTO"),
        ("results/HDFCBANK-2023-06-30.htm", "HDFCBANK"),
        ("annual/ITC-FY25.pdf", "ITC"),
    ],
)
def test_ticker_of(name: str, ticker: str) -> None:
    assert documents.ticker_of(name) == ticker


@pytest.mark.parametrize(
    ("name", "doc_id"),
    [
        ("annual/ITC-FY25.pdf", "verify-annual-ITC-FY25"),
        ("results/BAJAJ-AUTO-2025-09-30.pdf", "verify-results-BAJAJ-AUTO-2025-09-30"),
        ("results/HDFCBANK-2023-06-30.htm", "verify-results-HDFCBANK-2023-06-30"),
    ],
)
def test_doc_id_of_keeps_a_verify_document_apart_from_the_corpus(name: str, doc_id: str) -> None:
    assert documents.doc_id_of(name) == doc_id


def test_ticker_of_refuses_a_name_with_no_period() -> None:
    with pytest.raises(ValueError, match="no company and period"):
        documents.ticker_of("results/ITC.pdf")


def test_every_cross_checked_company_is_one_the_database_holds() -> None:
    assert set(VERIFY_TICKERS) <= set(FACTS_TICKERS), set(VERIFY_TICKERS) - set(FACTS_TICKERS)


def test_load_verify_document_refuses_an_entry_flagged_before_ocr_pages(verify_dir: Path) -> None:
    documents.flag_documents(verify_dir)
    manifest = documents.read_manifest(verify_dir)
    del manifest["results/TEXT-2025-03-31.pdf"]["ocr_pages"]
    documents.write_manifest(manifest, verify_dir)
    with pytest.raises(ValueError, match="not flagged"):
        documents.load_verify_document(
            "results/TEXT-2025-03-31.pdf", verify_dir, _pdf_reader, _html_reader
        )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(
            ("ROA", "0.48", "%"), ("ROA", "0.48%", ""), id="a percent cell after its figure"
        ),
        pytest.param(("ROA", "0.48", "", "%"), ("ROA", "0.48%", "", ""), id="empty cells between"),
        pytest.param(("ROA", "%", "0.48"), ("ROA", "%", "0.48"), id="a percent after a label"),
        pytest.param(("%", "0.48"), ("%", "0.48"), id="a percent with nothing before it"),
        pytest.param(("ROA", "0.48%"), ("ROA", "0.48%"), id="a figure with its own percent"),
        pytest.param(
            ("ROA", "0.48", "%", "0.47", "%"),
            ("ROA", "0.48%", "", "0.47%", ""),
            id="a percent cell after each figure",
        ),
    ],
)
def test_percent_joined(row: tuple[str, ...], expected: tuple[str, ...]) -> None:
    assert documents.percent_joined(row) == expected, row


_RESULTS = (
    ("FINANCIAL RESULTS", "", ""),
    ("Particulars", "March 31, 2024", "March 31, 2023"),
    ("Interest earned", "42,606.72", "34,438.91"),
    ("Exceptional items", "..", ".."),
)
_SHEET = (
    ("BALANCE SHEET", "", ""),
    ("(Rs. in crore)", "", ""),
    ("Particulars", "March 31, 2024", "March 31, 2023"),
    ("Deposits", "1,443,579.95", "1,210,832.15"),
)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(_RESULTS, [0], id="one statement"),
        pytest.param(_RESULTS + _SHEET, [0, 4], id="a second statement's title and headers"),
        pytest.param(_RESULTS + _SHEET[2:], [0, 4], id="a second header row with no title"),
        pytest.param(
            _RESULTS + (("Note", "as at March 31, 2024", ""),) + _SHEET[3:],
            [0],
            id="one date in a row of text",
        ),
        pytest.param(
            _RESULTS + (("", "", ""),) + _SHEET,
            [0, 5],
            id="an empty row is left with the statement before",
        ),
        pytest.param(
            (_SHEET[2],) + _SHEET[2:],
            [0],
            id="two header rows before any figure",
        ),
    ],
)
def test_statement_starts(rows: documents.Rows, expected: list[int]) -> None:
    assert documents.statement_starts(rows) == expected, rows


def test_statements_give_each_its_own_rows_headers_and_sections() -> None:
    parts = documents.statements(_RESULTS + _SHEET, header=2, sections=(3, 7))
    assert [p.rows for p in parts] == [_RESULTS, _SHEET]
    assert [p.header_rows for p in parts] == [2, 0]
    assert [p.sections for p in parts] == [(3,), (3,)]
    assert parts[1].text.splitlines()[0] == "BALANCE SHEET"


_STATEMENTS_HTML = """<html><body><table>
<tr><td>STANDALONE FINANCIAL RESULTS</td><td></td><td></td><td></td><td></td></tr>
<tr><td>Particulars</td><td>March 31, 2024</td><td></td><td>March 31, 2023</td><td></td></tr>
<tr><td>Return on assets</td><td>0.48</td><td>%</td><td>0.47</td><td>%</td></tr>
<tr><td>CONSOLIDATED BALANCE SHEET</td><td></td><td></td><td></td><td></td></tr>
<tr><td>Particulars</td><td>March 31, 2024</td><td></td><td>March 31, 2023</td><td></td></tr>
<tr><td>Deposits</td><td>1,443,579.95</td><td></td><td>1,210,832.15</td><td></td></tr>
</table></body></html>"""


def test_an_html_table_of_two_statements_reads_as_two(tmp_path: Path) -> None:
    path = tmp_path / "TWO-2024-03-31.htm"
    path.write_text(_STATEMENTS_HTML, encoding="utf-8")
    tables = documents.load_html(path, "TWO-2024-03-31").tables
    assert [t.rows[0][0] for t in tables] == [
        "STANDALONE FINANCIAL RESULTS",
        "CONSOLIDATED BALANCE SHEET",
    ]
    assert tables[0].rows[2] == ("Return on assets", "0.48%", "", "0.47%", ""), tables[0].rows


_TYPED: documents.Entry = {"pages": 10, "text_less_pages": [], "ocr_pages": []}
_OCRD: documents.Entry = {"pages": 10, "text_less_pages": [], "ocr_pages": list(range(2, 11))}


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param({"source": "nse", "format": "pdf"}, "nse-pdf", id="an nse pdf"),
        pytest.param({"source": "sec-6k", "format": "html"}, "sec-6k-html", id="an sec copy"),
        pytest.param({"source": "nse", "format": "pdf", **_TYPED}, "nse-pdf", id="a typed pdf"),
        pytest.param(
            {"source": "nse", "format": "pdf", **_OCRD}, "nse-pdf-ocr", id="a scan with OCR text"
        ),
    ],
)
def test_origin_names_source_and_form(entry: documents.Entry, expected: str) -> None:
    assert documents.origin(entry) == expected


_SCANNED: documents.Entry = {
    "source": "nse",
    "format": "pdf",
    "pages": 10,
    "text_less_pages": [],
    "ocr_pages": list(range(2, 11)),
}
_IMAGE_ONLY: documents.Entry = {**_SCANNED, "text_less_pages": list(range(2, 11)), "ocr_pages": []}
_TEXT_PDF: documents.Entry = {**_SCANNED, "ocr_pages": []}
_REOCR = "results/OCR-2025-03-31.pdf#rapidocr"


def test_stripping_a_scan_leaves_its_pages_and_no_text(tmp_path: Path) -> None:
    source = _pdf(tmp_path / "scan.pdf", [_TEXT] * 3, {2: [3], 3: [3]})
    removed = documents.strip_text_layer(source, tmp_path / "stripped.pdf")
    pages, text_less = documents.text_less_pages(tmp_path / "stripped.pdf")
    assert (removed, pages, text_less) == (3, 3, [1, 2, 3]), (removed, pages, text_less)


@pytest.mark.parametrize(
    ("name", "entry", "ocr_tickers", "readings"),
    [
        pytest.param(
            "results/OCR-2025-03-31.pdf", _SCANNED, ("OCR",), 2, id="a scan of a company read so"
        ),
        pytest.param(
            "results/OCR-2025-03-31.pdf", _IMAGE_ONLY, ("OCR",), 2, id="an image-only scan too"
        ),
        pytest.param(
            "results/OCR-2025-03-31.pdf", _SCANNED, ("TEXT",), 1, id="a scan of another company"
        ),
        pytest.param("results/OCR-2025-03-31.pdf", _TEXT_PDF, ("OCR",), 1, id="a text pdf"),
        pytest.param(
            "results/WEB-2023-06-30.htm",
            {
                "source": "sec-6k",
                "format": "html",
                "pages": 0,
                "text_less_pages": [],
                "ocr_pages": [],
            },
            ("WEB",),
            1,
            id="an html filing",
        ),
        pytest.param(
            "results/OCR-2025-03-31.pdf",
            {"source": "nse", "format": "pdf"},
            ("OCR",),
            1,
            id="an entry not flagged",
        ),
    ],
)
def test_readings_of(
    name: str, entry: documents.Entry, ocr_tickers: tuple[str, ...], readings: int
) -> None:
    got = documents.readings_of(name, entry, ocr_tickers)
    assert got == (name, f"{name}#rapidocr")[:readings], got


def test_a_reading_is_stamped_with_its_engine_and_version() -> None:
    assert documents.origin_of(_REOCR, _SCANNED) == f"nse-pdf-rapidocr-{version('rapidocr')}"
    assert documents.origin_of("results/OCR-2025-03-31.pdf", _SCANNED) == "nse-pdf-ocr"


def test_a_reading_has_its_own_name_and_reads_its_scan() -> None:
    assert documents.doc_id_of(_REOCR) == "verify-results-OCR-2025-03-31-rapidocr"
    assert documents.filed_of(_REOCR) == "results/OCR-2025-03-31.pdf"
    assert documents.is_reading(_REOCR) and not documents.is_reading(documents.filed_of(_REOCR))


def _reocr_reader(name: str, verify_dir: Path) -> Document:
    return Document(doc_id=f"reocr:{name}", text="", source=str(verify_dir))


@pytest.mark.parametrize(
    ("name", "ocr_tickers", "outcome"),
    [
        pytest.param(_REOCR, ("OCR",), "reocr:" + _REOCR, id="a scan's reading"),
        pytest.param(_REOCR, ("TEXT",), ValueError, id="a reading of a company not read so"),
        pytest.param(
            "results/TEXT-2025-03-31.pdf#rapidocr", ("TEXT",), ValueError, id="a text pdf's"
        ),
        pytest.param("results/NOPE.pdf#rapidocr", ("NOPE",), KeyError, id="a scan not held"),
    ],
)
def test_load_verify_document_reads_a_reading_through_reocr(
    verify_dir: Path, name: str, ocr_tickers: tuple[str, ...], outcome: str | type[Exception]
) -> None:
    documents.flag_documents(verify_dir)

    def load() -> Document:
        return documents.load_verify_document(
            name, verify_dir, _pdf_reader, _html_reader, ocr_tickers, _reocr_reader
        )

    if isinstance(outcome, str):
        assert load().doc_id == outcome
    else:
        with pytest.raises(outcome):
            load()


def test_a_reading_is_built_once_and_kept_under_its_fingerprint(
    verify_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    parsed: list[Path] = []

    def parse(path: Path) -> DoclingDocument:
        parsed.append(path)
        return docling([["Revenue 100 90"]])

    assert not documents.reocr_built(_REOCR, verify_dir, cache)
    with pytest.raises(ValueError, match="python -m aigent.facts reocr --documents"):
        documents.load_reocr(_REOCR, verify_dir, cache)
    kept = documents.build_reocr(_REOCR, verify_dir, cache, parse)
    read = documents.load_reocr(_REOCR, verify_dir, cache)
    assert documents.reocr_built(_REOCR, verify_dir, cache) and len(parsed) == 1
    assert read.doc_id == "verify-results-OCR-2025-03-31-rapidocr" and "Revenue" in read.text
    monkeypatch.setattr(documents, "_REOCR_FINGERPRINT", "0123456789ab")
    assert not documents.reocr_built(_REOCR, verify_dir, cache), "another engine's reading reused"
    documents.build_reocr(_REOCR, verify_dir, cache, parse)
    assert not kept.exists(), "the superseded reading is swept"


def test_a_reading_of_a_scan_not_held_is_never_built(tmp_path: Path) -> None:
    assert not documents.reocr_built("results/NOPE-2025-03-31.pdf#rapidocr", tmp_path, tmp_path)
