"""Annual report PDFs into the documents a chunker splits.

Docling per page, its text, headings and tables in reading order, each page normalised, then
joined while recording the offset each page begins at, so a citation still names the right page
after extraction has reflowed the text; each table's span and cells are kept beside it. Running
headers and page numbers go by how often they repeat, since no markup survives a PDF to say what is
furniture.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from functools import cache
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import cast

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.table.table_data import TableData
from docling_core.types.doc.items.text import SectionHeaderItem, TextItem, TitleItem
from pypdf import PdfWriter
from pypdf._codecs.adobe_glyphs import adobe_glyphs
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    StreamObject,
)

from aigent.config import (
    DOCLING_DEVICE,
    DOCLING_OCR,
    FURNITURE_EDGE_LINES,
    FURNITURE_MIN_PAGES,
    FURNITURE_RATIO,
    PAGE_SEPARATOR,
    PDF_RUPEE_KEYS,
)
from aigent.retrieval.chunk import Document, Heading, Table

REPO = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO / "corpus"
CACHE_DIR = CORPUS_DIR / ".cache"
TRAINING_DIR = CORPUS_DIR / "training"  # the fine-tune's reports; load_corpus never reads it
MANIFEST = REPO / "evals" / "reference" / "corpus-manifest.json"
TRAINING_MANIFEST = MANIFEST.with_name("training-manifest.json")  # never retrieved over
# The reports the published retrieval, RAG and agent numbers were measured over; the capstone reads
# the FY24 reports beside them.
FY25_REPORTS = ("ITC-FY25", "RELIANCE-FY25", "TATAMOTORS-FY25")

_HYPHENATED_BREAK = re.compile(r"(\w)-\n(\w)")
_RUN_OF_SPACES = re.compile(r" {2,}")
_RUN_OF_BLANK_LINES = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class Element:
    """One piece of a page as the parser laid it out: a block of text, a heading, or a table with
    its cells row by row."""

    text: str
    rows: tuple[tuple[str, ...], ...] | None = None  # None for text and headings
    heading: bool = False
    header_rows: int = 0  # a table's leading rows of column headers

    @property
    def is_table(self) -> bool:
        return self.rows is not None


@cache
def _converter() -> DocumentConverter:
    options = PdfPipelineOptions(do_ocr=DOCLING_OCR, do_table_structure=True)
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice(DOCLING_DEVICE))
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


_BFCHAR_BLOCK = re.compile(rb"beginbfchar(.*?)endbfchar", re.DOTALL)
_BFCHAR = re.compile(rb"(<[0-9A-Fa-f]+>\s*)<([0-9A-Fa-f]{4})>")
_RUPEE = "\u20b9"


def _family(font: DictionaryObject) -> str:
    """A font's family: its name without the subset tag or the style, `ABCDEF+IndianRupee-Regular`
    being `IndianRupee`."""
    return str(font.get("/BaseFont", "")).lstrip("/").split("+")[-1].split("-")[0]


def _rupee_entries(cmap: bytes, keys: str) -> bytes:
    """A ToUnicode map with every single-character entry reading as one of `keys` read as ₹."""

    def entry(match: re.Match[bytes]) -> bytes:
        read = chr(int(match.group(2), 16))
        return match.group(1) + (b"<20B9>" if read in keys else b"<" + match.group(2) + b">")

    return _BFCHAR_BLOCK.sub(lambda block: _BFCHAR.sub(entry, block.group(0)), cmap)


def _simple_font_map(font: DictionaryObject, keys: str) -> bytes | None:
    """A ToUnicode map for a one-byte WinAnsi font, each code read as its encoding reads it and the
    codes for `keys` as ₹; None for any other encoding."""
    encoding = font.get("/Encoding")
    encoding = encoding.get_object() if encoding is not None else None
    differences: list[object] = []
    if isinstance(encoding, DictionaryObject):
        differences = list(cast(ArrayObject, encoding.get("/Differences", ArrayObject())))
        encoding = encoding.get("/BaseEncoding")
    if encoding != "/WinAnsiEncoding":
        return None
    read: dict[int, str] = {}
    for code in range(0x20, 0x100):
        try:
            read[code] = bytes([code]).decode("cp1252")
        except UnicodeDecodeError:
            continue
    code = 0
    for item in differences:
        if isinstance(item, int):
            code = item
            continue
        glyph = adobe_glyphs.get(str(item))
        if glyph is None:
            read.pop(code, None)
        else:
            read[code] = glyph
        code += 1
    entries = [
        f"<{code:02X}> <{ord(_RUPEE if text in keys else text):04X}>"
        for code, text in sorted(read.items())
        if len(text) == 1
    ]
    blocks = (entries[i : i + 100] for i in range(0, len(entries), 100))
    return (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
        "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        + "".join(
            f"{len(block)} beginbfchar\n" + "\n".join(block) + "\nendbfchar\n" for block in blocks
        )
        + "endcmap CMapName currentdict /CMap defineresource pop end end\n"
    ).encode()


def rupees_decoded(path: Path) -> Path | DocumentStream:
    """The PDF in memory with the keys `PDF_RUPEE_KEYS` names read as ₹ in their fonts; the path
    itself if no font needed it."""
    writer = PdfWriter(clone_from=path)
    changed = False
    for page in writer.pages:
        resources = page.get("/Resources")
        fonts = resources.get_object().get("/Font") if resources else None
        for ref in (fonts.get_object() if fonts else DictionaryObject()).values():
            font = cast(DictionaryObject, ref.get_object())
            keys = PDF_RUPEE_KEYS.get(_family(font))
            if keys is None:
                continue
            if "/ToUnicode" in font:
                stream = cast(StreamObject, font["/ToUnicode"].get_object())
                before = stream.get_data()
                after = _rupee_entries(before, keys)
                if after != before:
                    stream.set_data(after)
                    changed = True
            elif (cmap := _simple_font_map(font, keys)) is not None:
                stream = DecodedStreamObject()
                stream.set_data(cmap)
                font[NameObject("/ToUnicode")] = writer._add_object(stream)  # no public way in
                changed = True
    if not changed:
        return path
    out = BytesIO()
    writer.write(out)
    out.seek(0)
    return DocumentStream(name=path.name, stream=out)


def convert(path: Path) -> DoclingDocument:
    """A PDF as Docling reads it, ₹ restored. The only function here that touches the parser."""
    return _converter().convert(rupees_decoded(path)).document


def table_rows(data: TableData) -> tuple[tuple[str, ...], ...]:
    """A table's cells row by row, a cell spanning several columns given once, in its first."""
    rows: list[tuple[str, ...]] = []
    for row in data.grid:
        cells: list[str] = []
        for column, cell in enumerate(row):
            first = cell.start_col_offset_idx == column or cell.start_col_offset_idx < 0
            cells.append(" ".join(cell.text.split()) if first else "")
        rows.append(tuple(cells))
    return tuple(rows)


def header_rows(data: TableData) -> int:
    """How many of a table's leading rows Docling marked as column headers."""
    marked = {
        row
        for cell in data.table_cells
        if cell.column_header
        for row in range(cell.start_row_offset_idx, cell.end_row_offset_idx)
    }
    count = 0
    while count in marked:
        count += 1
    return count


def elements_of(document: DoclingDocument) -> list[list[Element]]:
    """Every page of a Docling document as its elements in reading order, text inside a picture
    included, a table printed one row a line, its filled cells joined by spaces. A page with
    nothing on it comes back with no elements rather than missing (invariant 19).
    """
    pages: list[list[Element]] = [[] for _ in range(max(document.pages, default=0))]
    for item, _ in document.iterate_items(traverse_pictures=True):
        if not isinstance(item, DocItem) or not item.prov:
            continue
        page = pages[item.prov[0].page_no - 1]
        if isinstance(item, TableItem):
            rows = table_rows(item.data)
            text = "\n".join(" ".join(c for c in row if c) for row in rows if any(row))
            page.append(Element(text, rows, header_rows=header_rows(item.data)))
        elif isinstance(item, (SectionHeaderItem, TitleItem)):
            page.append(Element(item.text, heading=True))
        elif isinstance(item, TextItem) and item.text.strip():
            page.append(Element(item.text))
    return pages


def read_elements(path: Path) -> list[list[Element]]:
    """Every page of a PDF as its elements, as Docling lays them out."""
    return elements_of(convert(path))


def read_pages(path: Path) -> list[str]:
    """Raw extracted text, one string per page: each page's elements, one after another."""
    return ["\n".join(element.text for element in page) for page in read_elements(path)]


def normalise(text: str) -> str:
    """One page of extracted text, cleaned without dropping a character a quote could land on.

    Ligatures and exotic spaces fold to ASCII, a word hyphenated across a line break is rejoined,
    trailing spaces go, and runs of blank lines collapse to one.
    """
    text = unicodedata.normalize("NFKC", text).replace("\x0c", "\n")
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")
    text = _RUN_OF_SPACES.sub(" ", text.replace("\t", " "))
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _HYPHENATED_BREAK.sub(r"\1\2", text)
    return _RUN_OF_BLANK_LINES.sub("\n\n", text).strip()


def _skeleton(line: str) -> str:
    """A line with its digits masked — how two printings of one running header compare."""
    return _DIGITS.sub("#", line.strip())


def furniture(
    pages: Sequence[str],
    *,
    min_pages: int = FURNITURE_MIN_PAGES,
    edge_lines: int = FURNITURE_EDGE_LINES,
    ratio: float = FURNITURE_RATIO,
) -> set[str]:
    """The running headers, footers and page numbers a PDF repeats, as digit-masked skeletons.

    Only the first and last `edge_lines` of a page are candidates, so a label repeated mid-page is
    content. Candidates are counted with their digits masked, since a header carries the page
    number and so never repeats verbatim; one reaching `ratio` of the pages is furniture. Documents
    of fewer than `min_pages` pages have none.
    """
    if len(pages) < min_pages:
        return set()
    seen: Counter[str] = Counter()
    for page in pages:
        content = [line for line in page.split("\n") if line.strip()]
        edges = content[:edge_lines] + content[-edge_lines:]
        seen.update({_skeleton(line) for line in edges})
    return {line for line, pages_with_it in seen.items() if pages_with_it >= ratio * len(pages)}


def _unfurnished(text: str, found: set[str]) -> str:
    return "\n".join(
        line for line in text.split("\n") if not line.strip() or _skeleton(line) not in found
    ).strip()


def strip_furniture(
    pages: Sequence[str],
    *,
    min_pages: int = FURNITURE_MIN_PAGES,
    edge_lines: int = FURNITURE_EDGE_LINES,
    ratio: float = FURNITURE_RATIO,
) -> list[str]:
    """Pages with their `furniture` gone, wherever on a page it appears. A page that was nothing
    but furniture stays, empty, or every citation after it shifts by one."""
    found = furniture(pages, min_pages=min_pages, edge_lines=edge_lines, ratio=ratio)
    return [_unfurnished(page, found) for page in pages]


def to_document(
    doc_id: str,
    pages: Sequence[str],
    source: str = "",
    separator: str = PAGE_SEPARATOR,
) -> Document:
    """Join pages into one text, recording the offset each one starts at.

    `page_starts` indexes the joined text and is what puts a page number on a citation. An empty
    page still takes its place, or every page after it is cited one too low.
    """
    starts: list[int] = []
    offset = 0
    for index, page in enumerate(pages):
        if index:
            offset += len(separator)
        starts.append(offset)
        offset += len(page)
    return Document(
        doc_id=doc_id,
        text=separator.join(pages),
        source=source,
        page_starts=tuple(starts),
    )


def assembled(doc_id: str, elements: Sequence[Sequence[Element]], source: str = "") -> Document:
    """A document from its pages' elements: each normalised, the furniture stripped from its text
    but never from a heading or a table, and each table's and heading's span recorded as the
    pages are joined. An empty page still takes its place, or every page after it is cited low."""
    cleaned = [[replace(e, text=normalise(e.text)) for e in page] for page in elements]
    found = furniture(["\n".join(e.text for e in page) for page in cleaned])
    pages: list[str] = []
    spans: list[list[tuple[int, int, Element]]] = []
    for page in cleaned:
        parts: list[str] = []
        page_spans: list[tuple[int, int, Element]] = []
        offset = 0
        for element in page:
            text = (
                element.text
                if element.heading or element.is_table
                else _unfurnished(element.text, found)
            )
            if not text:
                continue
            if parts:
                offset += 1  # the newline joining two elements
            page_spans.append((offset, offset + len(text), element))
            parts.append(text)
            offset += len(text)
        pages.append("\n".join(parts))
        spans.append(page_spans)
    document = to_document(doc_id, pages, source=source)
    tables = tuple(
        Table(start + a, start + b, e.rows, e.header_rows)
        for start, page_spans in zip(document.page_starts, spans, strict=True)
        for a, b, e in page_spans
        if e.rows is not None
    )
    headings = tuple(
        Heading(start + a, start + b)
        for start, page_spans in zip(document.page_starts, spans, strict=True)
        for a, b, e in page_spans
        if e.heading
    )
    return replace(document, tables=tables, headings=headings)


def load_pdf(path: Path, doc_id: str | None = None) -> Document:
    """One PDF as a Document: read its elements, then `assembled`.

    `doc_id` defaults to the filename stem, which is what every chunk id is then built from.
    """
    return assembled(doc_id or path.stem, read_elements(path), source=str(path))


# Everything that decides what Docling reads a PDF as: hashed, so editing it re-parses on its own.
_PARSE_FINGERPRINT = hashlib.sha256(
    (
        "".join(
            inspect.getsource(function)
            for function in (
                _converter,
                _family,
                _rupee_entries,
                _simple_font_map,
                rupees_decoded,
                convert,
            )
        )
        + f"docling {version('docling')}|{DOCLING_OCR}|{sorted(PDF_RUPEE_KEYS.items())}"
    ).encode()
).hexdigest()[:12]

# Everything that decides what a PDF turns into: the parse, then the code rendering it as text.
_INGEST_FINGERPRINT = hashlib.sha256(
    (
        _PARSE_FINGERPRINT
        + "".join(
            inspect.getsource(function)
            for function in (
                table_rows,
                header_rows,
                elements_of,
                normalise,
                _skeleton,
                furniture,
                _unfurnished,
                to_document,
                assembled,
            )
        )
        + f"{FURNITURE_MIN_PAGES}|{FURNITURE_EDGE_LINES}|{FURNITURE_RATIO}|{PAGE_SEPARATOR!r}"
    ).encode()
).hexdigest()[:12]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def parse_file(path: Path, doc_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Where Docling's own reading of this PDF is kept, keyed by its bytes and the parse alone."""
    return cache_dir / f"{doc_id}.{_digest(path)}-{_PARSE_FINGERPRINT}.docling"


def parse_cached(path: Path, doc_id: str, cache_dir: Path = CACHE_DIR) -> DoclingDocument:
    """`convert`, memoised on disk, so a change to rendering never parses a PDF again."""
    kept = parse_file(path, doc_id, cache_dir)
    if kept.exists():
        try:
            return DoclingDocument.load_from_json(kept)
        except ValueError:
            kept.unlink(missing_ok=True)
    document = convert(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob(f"{doc_id}.*.docling"):
        stale.unlink(missing_ok=True)
    partial = kept.with_suffix(".partial")
    document.save_as_json(partial, indent=0)
    partial.replace(kept)
    return document


def cache_file(path: Path, doc_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Where this PDF's extracted text is kept, keyed by its bytes and by the ingestion code.

    Both halves matter. A company republishing a report changes the bytes; editing `normalise`
    changes the fingerprint. Either one makes this a different filename, so stale text is never
    read back — the cost of being wrong here is silent, and the cost of a spare re-parse is 30s.
    """
    return cache_dir / f"{doc_id}-{_digest(path)}-{_INGEST_FINGERPRINT}.json"


def load_pdf_cached(path: Path, doc_id: str | None = None, cache_dir: Path = CACHE_DIR) -> Document:
    """`load_pdf`, memoised on disk twice: the rendered document, and Docling's parse beneath it
    (`parse_cached`)."""
    name = doc_id or path.stem
    cached = cache_file(path, name, cache_dir)
    if cached.exists():
        try:
            stored = json.loads(cached.read_text(encoding="utf-8"))
            return Document(
                doc_id=stored["doc_id"],
                text=stored["text"],
                source=stored["source"],
                page_starts=tuple(stored["page_starts"]),
                tables=tuple(
                    Table(
                        t["start"],
                        t["end"],
                        tuple(tuple(row) for row in t["rows"]),
                        t["header_rows"],
                    )
                    for t in stored["tables"]
                ),
                headings=tuple(Heading(h["start"], h["end"]) for h in stored["headings"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            cached.unlink(missing_ok=True)

    parsed = parse_cached(path, name, cache_dir)
    document = assembled(name, elements_of(parsed), source=str(path))
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob(f"{name}-*.json"):
        stale.unlink(missing_ok=True)
    # Written aside and moved, so an interrupted run leaves no half-file to be read back as text.
    partial = cached.with_suffix(".partial")
    partial.write_text(
        json.dumps(
            {
                "doc_id": document.doc_id,
                "text": document.text,
                "source": document.source,
                "page_starts": list(document.page_starts),
                "tables": [
                    {
                        "start": t.start,
                        "end": t.end,
                        "rows": [list(row) for row in t.rows],
                        "header_rows": t.header_rows,
                    }
                    for t in document.tables
                ],
                "headings": [{"start": h.start, "end": h.end} for h in document.headings],
            }
        ),
        encoding="utf-8",
    )
    partial.replace(cached)
    return document


def load_corpus(
    directory: Path = CORPUS_DIR, *, cache: bool = False, doc_ids: Collection[str] | None = None
) -> list[Document]:
    """Every PDF directly in a directory, or just the `doc_ids` named, in filename order, so two
    runs build the same chunk ids. A named report that is not there raises.

    A directory with no PDFs raises rather than returning nothing: an empty corpus scores zero
    recall on every question and looks like a bad retriever.

    `cache` is opt-in. Reading the PDFs is always right; a cache is right until it is not, and a
    stale hit is wrong quietly. Ask for it where the 28 seconds is worth that, not everywhere.
    """
    paths = sorted(directory.glob("*.pdf"))
    if not paths:
        raise ValueError(f"no PDFs in {directory} — run ./scripts/fetch-corpus.sh to download them")
    if doc_ids is not None:
        missing = sorted(set(doc_ids) - {path.stem for path in paths})
        if missing:
            raise ValueError(f"{missing} not in {directory} — run ./scripts/fetch-corpus.sh")
        paths = [path for path in paths if path.stem in doc_ids]
    read = load_pdf_cached if cache else load_pdf
    return [read(path) for path in paths]
