"""The corpus: a PDF in, a `Document` with page offsets out.

The load-bearing table is the one on page offsets. Every citation downstream names a page, and
offsets taken before normalisation are wrong in a way nothing else in the pipeline notices.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigent.config import PAGE_SEPARATOR
from aigent.retrieval import corpus
from aigent.retrieval.corpus import (
    cache_file,
    load_corpus,
    load_pdf,
    load_pdf_cached,
    normalise,
    read_pages,
    strip_furniture,
    to_document,
)

HEADER = "RELIANCE INDUSTRIES LIMITED | INTEGRATED ANNUAL REPORT 2024-25"


def build_pdf(pages: list[list[str]]) -> bytes:
    """An uncompressed PDF with these lines on these pages, so a test can read one back."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog, pages_obj = add(b""), add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    kids: list[int] = []
    for lines in pages:
        text = b"BT /F1 12 Tf 72 720 Td 14 TL\n"
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            text += b"(" + escaped.encode("latin-1") + b") Tj T*\n"
        text += b"ET"
        content = add(
            b"<< /Length " + str(len(text)).encode() + b" >>\nstream\n" + text + b"\nendstream"
        )
        kids.append(
            add(
                b"<< /Type /Page /Parent " + str(pages_obj).encode() + b" 0 R "
                b"/MediaBox [0 0 612 792] /Contents " + str(content).encode() + b" 0 R "
                b"/Resources << /Font << /F1 " + str(font).encode() + b" 0 R >> >> >>"
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


def test_a_corpus_with_no_pdfs_raises_rather_than_scoring_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no PDFs"):
        load_corpus(tmp_path)


def test_read_pages_returns_one_string_per_page(tmp_path: Path) -> None:
    pages = read_pages(annual_report(tmp_path))

    assert len(pages) == 6
    assert "Revenue from operations" in pages[0]
    assert "3,89,000 people" in pages[5]


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


def _counting_reader(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace `read_pages` with one that records how often the PDF was really parsed."""
    calls = [0]
    original = corpus.read_pages

    def counted(path: Path) -> list[str]:
        calls[0] += 1
        return original(path)

    monkeypatch.setattr(corpus, "read_pages", counted)
    return calls


def test_a_second_load_is_served_from_cache_without_parsing_the_pdf_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = annual_report(tmp_path)
    calls = _counting_reader(monkeypatch)

    first = load_pdf_cached(path, cache_dir=tmp_path / ".cache")
    second = load_pdf_cached(path, cache_dir=tmp_path / ".cache")

    assert calls[0] == 1, "the second load re-parsed the PDF"
    assert second == first, "a cached document must be indistinguishable from a parsed one"
    assert second.page_starts == first.page_starts, "page offsets have to survive the round trip"


def test_republishing_the_report_invalidates_its_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A company quietly reissuing a report is the case that would poison a filename-keyed cache."""
    path = annual_report(tmp_path)
    calls = _counting_reader(monkeypatch)
    load_pdf_cached(path, cache_dir=tmp_path / ".cache")

    path.write_bytes(_revised_report())
    reloaded = load_pdf_cached(path, cache_dir=tmp_path / ".cache")

    assert calls[0] == 2, "changed bytes must not be served from the old cache"
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
    written = next(cache.glob("*.json"))
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
