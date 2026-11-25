"""The filed results and annual reports the cross-check reads XBRL against, under `corpus/verify/`:
each one's format, text-less pages and OCR'd pages kept in its manifest, and each read as a
`Document`."""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import cast

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import SectionHeaderItem, TextItem, TitleItem

from aigent.config import DOCLING_DEVICE, DOCLING_QUEUE_MAX_SIZE
from aigent.facts import line_rows
from aigent.facts.cfg_facts import (
    VERIFY_LINE_GAP,
    VERIFY_MAX_TEXT_LESS_SHARE,
    VERIFY_MIN_TEXT_CHARS,
    VERIFY_OCR_TICKERS,
    VERIFY_REOCR,
)
from aigent.retrieval import corpus
from aigent.retrieval.chunk import Document, Table
from aigent.retrieval.corpus import (
    CACHE_DIR,
    CORPUS_DIR,
    Element,
    StoredDocument,
    assembled,
    captions_restored,
    document_from_json,
    elements_of,
    header_rows,
    parse_cached,
    rupees_decoded,
    section_rows,
    table_rows,
    write_cached,
)

VERIFY_DIR = CORPUS_DIR / "verify"
MANIFEST_NAME = "manifest.json"
FORMATS = {".htm": "html", ".html": "html", ".pdf": "pdf"}
REOCR_MARK = f"#{VERIFY_REOCR}"  # a scan's re-OCR'd reading: results/X-2025-03-31.pdf#rapidocr
_FLAGGED = {"format", "pages", "text_less_pages", "ocr_pages"}

Entry = dict[str, object]
Rows = tuple[tuple[str, ...], ...]

_FIGURE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?%?|\.\.|[-–—]")  # a nil prints ".." or a dash
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(
    rf"\b{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}|\b\d{{1,2}}\s+{_MONTH},?\s+\d{{4}}"
    r"|\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
    re.IGNORECASE,
)


def text_less_pages(path: Path, min_chars: int = VERIFY_MIN_TEXT_CHARS) -> tuple[int, list[int]]:
    """The PDF's page count, and the 1-based pages whose text layer holds fewer than `min_chars`
    characters."""
    document = pdfium.PdfDocument(str(path))
    try:
        return len(document), [
            number
            for number, page in enumerate(document, 1)
            if len(page.get_textpage().get_text_range().strip()) < min_chars
        ]
    finally:
        document.close()


def ocr_pages(path: Path) -> list[int]:
    """The 1-based pages whose text is mostly drawn invisible: a scan with OCR text behind it."""
    document = pdfium.PdfDocument(str(path))
    try:
        found: list[int] = []
        for number, page in enumerate(document, 1):
            modes = [
                pdfium_c.FPDFTextObj_GetTextRenderMode(text.raw)
                for text in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT])
            ]
            invisible = sum(mode == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE for mode in modes)
            if modes and invisible * 2 > len(modes):
                found.append(number)
        return found
    finally:
        document.close()


def is_scan(entry: Entry, max_share: float = VERIFY_MAX_TEXT_LESS_SHARE) -> bool:
    """A flagged PDF entry with more than `max_share` of its pages text-less or OCR'd."""
    pages = cast(int, entry["pages"])
    scanned = set(cast(list[int], entry["text_less_pages"]))
    scanned |= set(cast(list[int], entry["ocr_pages"]))
    return pages > 0 and len(scanned) > max_share * pages


def is_ocr_scan(entry: Entry, max_share: float = VERIFY_MAX_TEXT_LESS_SHARE) -> bool:
    """A scan whose pages carry OCR text: over the share once OCR'd pages count, within it on its
    text-less pages alone."""
    pages = cast(int, entry["pages"])
    return (
        is_scan(entry, max_share)
        and len(cast(list[int], entry["text_less_pages"])) <= max_share * pages
    )


def origin(entry: Entry) -> str:
    """Where a flagged entry's copy came from, and its form: `nse-pdf`, `sec-6k-html`, and
    `nse-pdf-ocr` for a scan read through its OCR text."""
    form = f"{entry['source']}-{entry['format']}"
    return f"{form}-ocr" if "pages" in entry and is_ocr_scan(entry) else form


def ticker_of(name: str) -> str:
    """The company a manifest path names, as in `results/BAJAJ-AUTO-2025-09-30.pdf` or
    `annual/ITC-FY25.pdf`."""
    found = re.fullmatch(r"(.+)-(?:\d{4}-\d{2}-\d{2}|FY\d{2})", Path(name).stem)
    if found is None:
        raise ValueError(f"{name} names no company and period")
    return found.group(1)


def doc_id_of(name: str) -> str:
    """A manifest path's name in the parse cache, which holds one entry per name, so a corpus copy
    of the same report never evicts it: `annual/ITC-FY25.pdf` is `verify-annual-ITC-FY25`, its
    re-OCR'd reading's `verify-annual-ITC-FY25-rapidocr`."""
    if is_reading(name):
        return f"{doc_id_of(filed_of(name))}-{VERIFY_REOCR}"
    path = Path(name)
    return f"verify-{path.parent.name}-{path.stem}"


def is_reading(name: str) -> bool:
    """A re-OCR'd reading of a scan, rather than the filed document."""
    return name.endswith(REOCR_MARK)


def filed_of(name: str) -> str:
    """The filed document a name reads: itself, or the scan a reading re-reads."""
    return name.removesuffix(REOCR_MARK)


def readings_of(
    name: str, entry: Entry, ocr_tickers: Sequence[str] = VERIFY_OCR_TICKERS
) -> tuple[str, ...]:
    """An entry's readings: itself, and for a flagged scan of a company in `ocr_tickers`, its
    re-OCR'd reading `<name>#rapidocr`, image-only scans included."""
    scan = _FLAGGED <= entry.keys() and entry["format"] == "pdf" and is_scan(entry)
    return (name, name + REOCR_MARK) if scan and ticker_of(name) in ocr_tickers else (name,)


def origin_of(name: str, entry: Entry) -> str:
    """A reading's origin, the stamp's prefix: the entry's `origin`, or for a re-OCR'd reading its
    engine and version, `nse-pdf-rapidocr-3.10.0`."""
    if is_reading(name):
        return f"{entry['source']}-{entry['format']}-{VERIFY_REOCR}-{version(VERIFY_REOCR)}"
    return origin(entry)


def strip_text_layer(source: Path, target: Path) -> int:
    """Writes a copy of a PDF with every text object removed, its page images kept, so OCR
    supplies the words; Docling reads a table's words from the text layer before any OCR. Returns
    how many were removed."""
    document = pdfium.PdfDocument(str(source))
    try:
        removed = 0
        for number in range(len(document)):
            page = document[number]
            for text in list(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT])):
                page.remove_obj(text)
                text.close()  # removed, so ours to free
                removed += 1
            page.gen_content()
            page.close()
        document.save(str(target))
        return removed
    finally:
        document.close()


@cache
def _reocr_converter() -> DocumentConverter:
    ocr = RapidOcrOptions(backend="torch", mode=OcrMode.FULL_PAGE)
    options = PdfPipelineOptions(do_ocr=True, do_table_structure=True, ocr_options=ocr)
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice(DOCLING_DEVICE))
    options.queue_max_size = DOCLING_QUEUE_MAX_SIZE
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _reocr_parse(path: Path) -> DoclingDocument:
    """A scan as Docling reads it with its text layer removed and every page OCR'd."""
    with tempfile.TemporaryDirectory() as scratch:
        stripped = Path(scratch) / path.name
        strip_text_layer(path, stripped)
        result = _reocr_converter().convert(stripped)
    return captions_restored(result.document, {})  # no printed lines left to restore from


# Everything that decides what a re-OCR'd reading holds, so a change to any of it rebuilds them.
_REOCR_FINGERPRINT = hashlib.sha256(
    (
        "".join(inspect.getsource(f) for f in (strip_text_layer, _reocr_converter, _reocr_parse))
        + f"docling {version('docling')}|{VERIFY_REOCR} {version(VERIFY_REOCR)}"
        + f"|{corpus.ingest_fingerprint()}|{DOCLING_QUEUE_MAX_SIZE}"
    ).encode()
).hexdigest()[:12]


def reocr_file(name: str, verify_dir: Path = VERIFY_DIR, cache_dir: Path = CACHE_DIR) -> Path:
    """Where a re-OCR'd reading is kept, keyed by its scan's bytes and the re-OCR fingerprint."""
    digest = hashlib.sha256((verify_dir / filed_of(name)).read_bytes()).hexdigest()[:16]
    return cache_dir / f"{doc_id_of(name)}-{digest}-{_REOCR_FINGERPRINT}.json"


def build_reocr(
    name: str,
    verify_dir: Path = VERIFY_DIR,
    cache_dir: Path = CACHE_DIR,
    parse: Callable[[Path], DoclingDocument] = _reocr_parse,
) -> Path:
    """OCRs a scan afresh into its reading's cache file, sweeping superseded ones; the file."""
    path = verify_dir / filed_of(name)
    document = assembled(doc_id_of(name), elements_of(parse(path)), source=str(path))
    target = reocr_file(name, verify_dir, cache_dir)
    for stale in cache_dir.glob(f"{doc_id_of(name)}-*.json"):
        stale.unlink(missing_ok=True)
    write_cached(target, document)
    return target


def load_reocr(name: str, verify_dir: Path = VERIFY_DIR, cache_dir: Path = CACHE_DIR) -> Document:
    """A built re-OCR'd reading; ValueError naming the command that builds it when it is not."""
    kept = reocr_file(name, verify_dir, cache_dir)
    if not kept.exists():
        raise ValueError(
            f"{name} is not built: python -m aigent.facts reocr --documents {filed_of(name)}"
        )
    return document_from_json(cast(StoredDocument, json.loads(kept.read_text(encoding="utf-8"))))


def reocr_built(name: str, verify_dir: Path = VERIFY_DIR, cache_dir: Path = CACHE_DIR) -> bool:
    """Whether a reading's cache file is there: never for a scan not held."""
    held = (verify_dir / filed_of(name)).exists()
    return held and reocr_file(name, verify_dir, cache_dir).exists()


def refusal(name: str, entry: Entry, ocr_tickers: Sequence[str] = VERIFY_OCR_TICKERS) -> str | None:
    """Why a manifest entry cannot be read, or None: not flagged yet, or a scan, unless its pages
    carry OCR text and its company is in `ocr_tickers`."""
    if not {"format", "pages", "text_less_pages", "ocr_pages"} <= entry.keys():
        return f"{name} is not flagged yet; run flag_documents first"
    if entry["format"] == "html" or not is_scan(entry):
        return None
    if is_ocr_scan(entry) and ticker_of(name) in ocr_tickers:
        return None
    return f"{name} is a scan: no text layer on pages {entry['text_less_pages']}" + (
        f", OCR text on pages {entry['ocr_pages']}" if entry["ocr_pages"] else ""
    )


def read_manifest(verify_dir: Path = VERIFY_DIR) -> dict[str, Entry]:
    path = verify_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    return cast(dict[str, Entry], json.loads(path.read_text(encoding="utf-8")))


def write_manifest(manifest: dict[str, Entry], verify_dir: Path = VERIFY_DIR) -> None:
    """Written sorted by path, each entry's fields in one order."""
    ordered = {name: dict(sorted(manifest[name].items())) for name in sorted(manifest)}
    text = json.dumps(ordered, indent=1) + "\n"
    (verify_dir / MANIFEST_NAME).write_text(text, encoding="utf-8")


def flag_documents(
    verify_dir: Path = VERIFY_DIR,
    min_chars: int = VERIFY_MIN_TEXT_CHARS,
    max_share: float = VERIFY_MAX_TEXT_LESS_SHARE,
) -> list[str]:
    """Sets every entry's `format`, `pages`, `text_less_pages` and `ocr_pages` (0 and empty for
    HTML), and returns the scans, which `load_verify_document` refuses."""
    manifest = read_manifest(verify_dir)
    scanned: list[str] = []
    for name, entry in manifest.items():
        kind = FORMATS.get(Path(name).suffix.lower())
        if kind is None:
            raise ValueError(f"{name}: not one of {sorted(FORMATS)}")
        count, pages = text_less_pages(verify_dir / name, min_chars) if kind == "pdf" else (0, [])
        entry["format"] = kind
        entry["pages"] = count
        entry["text_less_pages"] = pages
        entry["ocr_pages"] = ocr_pages(verify_dir / name) if kind == "pdf" else []
        if is_scan(entry, max_share):
            scanned.append(name)
    write_manifest(manifest, verify_dir)
    return sorted(scanned)


def percent_joined(row: Sequence[str]) -> tuple[str, ...]:
    """A row with each cell holding only "%" joined onto the figure in the filled cell before it."""
    cells = list(row)
    for j, cell in enumerate(cells):
        if cell.strip() != "%":
            continue
        k = next((i for i in range(j - 1, -1, -1) if cells[i].strip()), None)
        if k is not None and _FIGURE.fullmatch(cells[k].strip()):
            cells[k] = cells[k].strip() + "%"
            cells[j] = ""
    return tuple(cells)


def statement_starts(rows: Rows) -> list[int]:
    """Where each statement starts in a table holding several: row 0, then the first of the text
    rows (title, scale, column headers) before each row of dates that follows figures."""

    def has_figure(row: Sequence[str]) -> bool:
        return any(_FIGURE.fullmatch(c.strip()) for c in row if c.strip())

    starts = [0]
    seen_figures = False
    for i, row in enumerate(rows):
        if has_figure(row):
            seen_figures = True
            continue
        if seen_figures and sum(bool(_DATE.search(c)) for c in row) >= 2:
            start = i
            while (
                start - 1 > starts[-1] and any(rows[start - 1]) and not has_figure(rows[start - 1])
            ):
                start -= 1
            starts.append(start)
            seen_figures = False
    return starts


def statements(rows: Rows, header: int, sections: tuple[int, ...]) -> list[Element]:
    """A table's rows as one element per statement, each with its own column headers."""
    starts = statement_starts(rows)
    out: list[Element] = []
    for n, (start, end) in enumerate(zip(starts, [*starts[1:], len(rows)], strict=True)):
        part = rows[start:end]
        out.append(
            Element(
                "\n".join(" ".join(c for c in row if c) for row in part if any(row)),
                part,
                header_rows=header if n == 0 else 0,
                sections=tuple(r - start for r in sections if start <= r < end),
            )
        )
    return out


def html_elements(document: DoclingDocument) -> list[list[Element]]:
    """A Docling reading of an HTML page as one page of elements in reading order, tables row by
    row, a lone "%" cell joined to its figure, and a table holding several statements split into
    one per statement; `retrieval.corpus.elements_of` keeps only items placed on a PDF page."""
    page: list[Element] = []
    for item, _ in document.iterate_items(traverse_pictures=True):
        if not isinstance(item, DocItem) or item.content_layer != ContentLayer.BODY:
            continue
        if isinstance(item, TableItem):
            rows = tuple(percent_joined(row) for row in table_rows(item.data))
            page.extend(statements(rows, header_rows(item.data), section_rows(item.data)))
        elif isinstance(item, (SectionHeaderItem, TitleItem)):
            page.append(Element(item.text, heading=True))
        elif isinstance(item, TextItem) and item.text.strip():
            page.append(Element(item.text))
    return [page]


def load_html(path: Path, doc_id: str) -> Document:
    """An HTML filing as a `Document`; Docling reads one in about a second, so it is not cached."""
    parsed = DocumentConverter().convert(path).document
    return assembled(doc_id, html_elements(parsed), source=str(path))


def _docling_tables(parsed: DoclingDocument) -> list[TableItem]:
    """Docling's tables in the order `elements_of` renders them: page by page, reading order."""
    attached = {
        ref.cref
        for item, _ in parsed.iterate_items()
        if isinstance(item, TableItem)
        for ref in (*item.captions, *item.footnotes)
    }
    layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
    found = [
        item
        for item, _ in parsed.iterate_items(traverse_pictures=True, included_content_layers=layers)
        if isinstance(item, TableItem)
        and item.prov
        and item.self_ref not in attached
        and item.content_layer != ContentLayer.FURNITURE
    ]
    return sorted(found, key=lambda item: item.prov[0].page_no)


def _slice_of(rows: Rows, body: Rows) -> int | None:
    """Where `body` sits inside a rendered table's rows, after its caption lines; None if not."""
    for start in range(len(rows) - len(body) + 1):
        if rows[start : start + len(body)] == body:
            return start
    return None


def with_line_rows(
    document: Document, parsed: DoclingDocument, words: Mapping[int, Sequence[line_rows.Word]]
) -> Document:
    """A rendered PDF with each table's body re-rowed from its printed lines (`line_rows`), given
    the words of each page with a table; its captions, footnotes and Docling's header rows kept,
    and a table it cannot re-row left as Docling read it. ValueError when a rendered table
    matches none of Docling's."""
    items = _docling_tables(parsed)
    tables: list[Table] = []
    k = 0
    for table in document.tables:
        while k < len(items) and _slice_of(table.rows, table_rows(items[k].data)) is None:
            k += 1
        if k == len(items):
            raise ValueError(f"{document.doc_id}: a rendered table matches none of Docling's")
        item, k = items[k], k + 1
        body = table_rows(item.data)
        start = _slice_of(table.rows, body) or 0
        page_no = item.prov[0].page_no
        shape = line_rows.layout(item, body, parsed.pages[page_no].size.height)
        rebuilt = (
            None
            if shape is None
            else line_rows.line_rows(body[: shape.header], shape, words.get(page_no, ()))
        )
        if shape is None or rebuilt is None:
            tables.append(table)
            continue
        kept = start + shape.header
        tables.append(
            replace(
                table,
                rows=table.rows[:start] + rebuilt + table.rows[start + len(body) :],
                header_rows=min(table.header_rows, kept),
                sections=tuple(s for s in table.sections if s < kept),
            )
        )
    return replace(document, tables=tuple(tables))


def _opened(path: Path) -> pdfium.PdfDocument:
    """A PDF opened with its rupee fonts read as ₹, as Docling read it."""
    decoded = rupees_decoded(path)
    return pdfium.PdfDocument(decoded if isinstance(decoded, Path) else decoded.stream.getvalue())


def read_words(path: Path, pages: Sequence[int]) -> dict[int, list[line_rows.Word]]:
    """The words of these 1-based pages of a PDF, placed on each page as displayed, boxes rounded
    to a hundredth of a point."""
    pdf = _opened(path)
    try:
        out: dict[int, list[line_rows.Word]] = {}
        for page_no in pages:
            page = pdf[page_no - 1]
            left, bottom, right, top = line_rows.crop_box(page)
            placed = line_rows.on_display(
                line_rows.page_words(page), page.get_rotation(), right - left, top - bottom
            )
            out[page_no] = [
                (text, (round(a, 2), round(b, 2), round(c, 2), round(d, 2)), spaced)
                for text, (a, b, c, d), spaced in placed
            ]
        return out
    finally:
        pdf.close()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# Everything that decides a filed PDF's words: rarely changed, so kept below the re-rowing.
_WORDS_FINGERPRINT = hashlib.sha256(
    (
        "".join(
            inspect.getsource(f)
            for f in (
                line_rows.split_runs,
                line_rows.page_words,
                line_rows.crop_box,
                line_rows.displayed,
                line_rows.on_display,
                _docling_tables,
                _opened,
                read_words,
            )
        )
        + f"|{corpus.parse_fingerprint()}|{line_rows.GRID}"
    ).encode()
).hexdigest()[:12]

# Everything that decides how a filed PDF's tables are re-rowed, above its words and rendering.
_LINES_FINGERPRINT = hashlib.sha256(
    (
        "".join(
            inspect.getsource(f)
            for f in (
                line_rows.layout,
                line_rows.line_rows,
                line_rows.joined,
                line_rows.column,
                _slice_of,
                with_line_rows,
            )
        )
        + f"|{line_rows.SEPARATOR.pattern}|{VERIFY_LINE_GAP}"
        + f"|{_WORDS_FINGERPRINT}|{corpus.ingest_fingerprint()}"
    ).encode()
).hexdigest()[:12]


def words_file(path: Path, doc_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Where a filed PDF's words are kept, keyed by its bytes and the words fingerprint."""
    return cache_dir / f"{doc_id}-words-{_digest(path)}-{_WORDS_FINGERPRINT}.json.gz"


def lines_file(path: Path, doc_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Where a filed PDF re-rowed is kept, keyed by its bytes and the re-rowing fingerprint."""
    return cache_dir / f"{doc_id}-lines-{_digest(path)}-{_LINES_FINGERPRINT}.json"


def load_words_cached(
    path: Path,
    doc_id: str,
    parsed: DoclingDocument,
    cache_dir: Path = CACHE_DIR,
    read: Callable[[Path, Sequence[int]], dict[int, list[line_rows.Word]]] = read_words,
) -> dict[int, list[line_rows.Word]]:
    """The words of each page of a filed PDF that Docling found a table on (`read_words`),
    memoised on disk, gzipped; a file that does not read back is rebuilt, superseded ones swept."""
    kept = words_file(path, doc_id, cache_dir)
    if kept.exists():
        try:
            stored = json.loads(gzip.decompress(kept.read_bytes()))
            return {
                int(page): [(str(t), (b[0], b[1], b[2], b[3]), bool(s)) for t, b, s in words]
                for page, words in stored.items()
            }
        except (OSError, ValueError, TypeError, IndexError):
            kept.unlink(missing_ok=True)
    pages = sorted({item.prov[0].page_no for item in _docling_tables(parsed)})
    words = read(path, pages)
    for stale in cache_dir.glob(f"{doc_id}-words-*.json.gz"):
        stale.unlink(missing_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = kept.with_suffix(".partial")
    partial.write_bytes(gzip.compress(json.dumps(words).encode()))
    partial.replace(kept)
    return words


def load_lines_cached(
    path: Path,
    doc_id: str,
    cache_dir: Path = CACHE_DIR,
    parsed: Callable[[Path, str, Path], DoclingDocument] = parse_cached,
    words: Callable[[Path, str, DoclingDocument, Path], dict[int, list[line_rows.Word]]] = (
        load_words_cached
    ),
) -> Document:
    """A filed PDF as the cross-check reads it: rendered from Docling's cached reading, its tables
    re-rowed from its cached words (`with_line_rows`), memoised on disk; a file that does not
    read back is rebuilt, superseded ones swept."""
    kept = lines_file(path, doc_id, cache_dir)
    if kept.exists():
        try:
            return document_from_json(cast(StoredDocument, json.loads(kept.read_text("utf-8"))))
        except (json.JSONDecodeError, KeyError, TypeError):
            kept.unlink(missing_ok=True)
    reading = parsed(path, doc_id, cache_dir)
    document = assembled(doc_id, elements_of(reading), source=str(path))
    rebuilt = with_line_rows(document, reading, words(path, doc_id, reading, cache_dir))
    for stale in cache_dir.glob(f"{doc_id}-lines-*.json"):
        stale.unlink(missing_ok=True)
    write_cached(kept, rebuilt)
    return rebuilt


def load_verify_document(
    name: str,
    verify_dir: Path = VERIFY_DIR,
    pdf: Callable[[Path, str], Document] = load_lines_cached,
    html: Callable[[Path, str], Document] = load_html,
    ocr_tickers: Sequence[str] = VERIFY_OCR_TICKERS,
    reocr: Callable[[str, Path], Document] = load_reocr,
) -> Document:
    """One manifest entry as a `Document`, read by the format its entry records, or a scan's
    re-OCR'd reading through `reocr`. Refuses a path not in the manifest (KeyError) and whatever
    `refusal` names (ValueError)."""
    entry = read_manifest(verify_dir).get(filed_of(name))
    if entry is None:
        raise KeyError(f"{filed_of(name)} is not in {verify_dir / MANIFEST_NAME}")
    if is_reading(name):
        if name not in readings_of(filed_of(name), entry, ocr_tickers):
            raise ValueError(
                f"{filed_of(name)} has no re-OCR'd reading: not a scan of {ocr_tickers}"
            )
        return reocr(name, verify_dir)
    refused = refusal(name, entry, ocr_tickers)
    if refused is not None:
        raise ValueError(refused)
    path = verify_dir / name
    if entry["format"] == "html":
        return html(path, doc_id_of(name))
    return pdf(path, doc_id_of(name))
