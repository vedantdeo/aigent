"""`facts.documents`: each filed document's format and text-less pages, read by its format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject

from aigent.facts import documents
from aigent.facts.cfg_facts import FACTS_TICKERS, VERIFY_TICKERS
from aigent.retrieval.chunk import Document

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


def _pdf(path: Path, pages: list[str]) -> Path:
    """A PDF with one page per string; an empty string is a page with no text layer."""
    writer = PdfWriter()
    for text in pages:
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
        stream.set_data(f"BT /F1 6 Tf 20 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = ArrayObject([writer._add_object(stream)])
    with path.open("wb") as out:
        writer.write(out)
    return path


@pytest.fixture
def verify_dir(tmp_path: Path) -> Path:
    """A verify folder: a text PDF, a scanned PDF, a report with one photo page, and HTML."""
    root = tmp_path / "verify"
    (root / "results").mkdir(parents=True)
    _pdf(root / "results/TEXT-2025-03-31.pdf", [_TEXT, _TEXT])
    _pdf(root / "results/SCAN-2025-03-31.pdf", [_TEXT, "", ""])
    _pdf(root / "results/PHOTO-FY25.pdf", [""] + [_TEXT] * 10)
    (root / "results/WEB-2023-06-30.htm").write_text(_HTML, encoding="utf-8")
    names = ("TEXT-2025-03-31.pdf", "SCAN-2025-03-31.pdf", "PHOTO-FY25.pdf", "WEB-2023-06-30.htm")
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
    ("pages", "text_less", "scan"),
    [
        pytest.param(10, [], False, id="no text-less page"),
        pytest.param(10, [1], False, id="exactly the share allowed"),
        pytest.param(10, [1, 2], True, id="over the share"),
        pytest.param(585, list(range(16)), False, id="an annual report's photo pages"),
        pytest.param(13, list(range(2, 14)), True, id="a scanned results filing"),
        pytest.param(0, [], False, id="html has no pages"),
    ],
)
def test_is_scan(pages: int, text_less: list[int], scan: bool) -> None:
    entry: documents.Entry = {"pages": pages, "text_less_pages": text_less}
    assert documents.is_scan(entry) is scan, entry


@pytest.mark.parametrize(
    ("name", "kind", "count", "pages"),
    [
        pytest.param("results/TEXT-2025-03-31.pdf", "pdf", 2, [], id="a text pdf"),
        pytest.param("results/SCAN-2025-03-31.pdf", "pdf", 3, [2, 3], id="a scanned pdf"),
        pytest.param("results/PHOTO-FY25.pdf", "pdf", 11, [1], id="a report with a photo page"),
        pytest.param("results/WEB-2023-06-30.htm", "html", 0, [], id="an html filing"),
    ],
)
def test_flag_documents_records_each_entry(
    verify_dir: Path, name: str, kind: str, count: int, pages: list[int]
) -> None:
    documents.flag_documents(verify_dir)
    entry = documents.read_manifest(verify_dir)[name]
    assert entry["format"] == kind, entry
    assert entry["pages"] == count, entry
    assert entry["text_less_pages"] == pages, entry
    assert entry["url"] == f"https://x/{name.removeprefix('results/')}", entry


def test_flag_documents_returns_the_scans(verify_dir: Path) -> None:
    assert documents.flag_documents(verify_dir) == ["results/SCAN-2025-03-31.pdf"]


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
    ("name", "doc_id"),
    [
        pytest.param("results/TEXT-2025-03-31.pdf", "pdf:TEXT-2025-03-31", id="a text pdf"),
        pytest.param("results/PHOTO-FY25.pdf", "pdf:PHOTO-FY25", id="a few photo pages pass"),
        pytest.param("results/WEB-2023-06-30.htm", "html:WEB-2023-06-30", id="an html filing"),
    ],
)
def test_load_verify_document_reads_by_format(verify_dir: Path, name: str, doc_id: str) -> None:
    documents.flag_documents(verify_dir)
    read = documents.load_verify_document(name, verify_dir, _pdf_reader, _html_reader)
    assert read.doc_id == doc_id, read.doc_id


@pytest.mark.parametrize(
    ("flag_first", "name", "error"),
    [
        pytest.param(True, "results/NOPE.pdf", KeyError, id="a path not in the manifest"),
        pytest.param(False, "results/TEXT-2025-03-31.pdf", ValueError, id="an entry not flagged"),
        pytest.param(True, "results/SCAN-2025-03-31.pdf", ValueError, id="a scanned pdf"),
    ],
)
def test_load_verify_document_refuses(
    verify_dir: Path, flag_first: bool, name: str, error: type[Exception]
) -> None:
    if flag_first:
        documents.flag_documents(verify_dir)
    with pytest.raises(error):
        documents.load_verify_document(name, verify_dir, _pdf_reader, _html_reader)


def test_every_cross_checked_company_is_one_the_database_holds() -> None:
    assert set(VERIFY_TICKERS) <= set(FACTS_TICKERS), set(VERIFY_TICKERS) - set(FACTS_TICKERS)
