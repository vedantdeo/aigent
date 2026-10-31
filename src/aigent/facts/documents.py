"""The filed results and annual reports the cross-check reads XBRL against, under `corpus/verify/`:
each one's format and text-less pages kept in its manifest, and each read as a `Document`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pypdfium2 as pdfium
from docling.document_converter import DocumentConverter
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import SectionHeaderItem, TextItem, TitleItem

from aigent.facts.cfg_facts import VERIFY_MAX_TEXT_LESS_SHARE, VERIFY_MIN_TEXT_CHARS
from aigent.retrieval.chunk import Document
from aigent.retrieval.corpus import (
    CORPUS_DIR,
    Element,
    assembled,
    header_rows,
    load_pdf_cached,
    section_rows,
    table_rows,
)

VERIFY_DIR = CORPUS_DIR / "verify"
MANIFEST_NAME = "manifest.json"
FORMATS = {".htm": "html", ".html": "html", ".pdf": "pdf"}

Entry = dict[str, object]


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


def is_scan(entry: Entry, max_share: float = VERIFY_MAX_TEXT_LESS_SHARE) -> bool:
    """A flagged PDF entry with more than `max_share` of its pages text-less."""
    pages = cast(int, entry["pages"])
    return pages > 0 and len(cast(list[int], entry["text_less_pages"])) > max_share * pages


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
    """Sets every entry's `format`, `pages` and `text_less_pages` (0 and empty for HTML), and
    returns the scans, which `load_verify_document` refuses."""
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
        if is_scan(entry, max_share):
            scanned.append(name)
    write_manifest(manifest, verify_dir)
    return sorted(scanned)


def html_elements(document: DoclingDocument) -> list[list[Element]]:
    """A Docling reading of an HTML page as one page of elements in reading order, tables row by
    row; `retrieval.corpus.elements_of` keeps only items placed on a PDF page, so drops them all."""
    page: list[Element] = []
    for item, _ in document.iterate_items(traverse_pictures=True):
        if not isinstance(item, DocItem) or item.content_layer != ContentLayer.BODY:
            continue
        if isinstance(item, TableItem):
            rows = table_rows(item.data)
            text = "\n".join(" ".join(c for c in row if c) for row in rows if any(row))
            page.append(
                Element(
                    text,
                    rows,
                    header_rows=header_rows(item.data),
                    sections=section_rows(item.data),
                )
            )
        elif isinstance(item, (SectionHeaderItem, TitleItem)):
            page.append(Element(item.text, heading=True))
        elif isinstance(item, TextItem) and item.text.strip():
            page.append(Element(item.text))
    return [page]


def load_html(path: Path, doc_id: str) -> Document:
    """An HTML filing as a `Document`; Docling reads one in about a second, so it is not cached."""
    parsed = DocumentConverter().convert(path).document
    return assembled(doc_id, html_elements(parsed), source=str(path))


def load_verify_document(
    name: str,
    verify_dir: Path = VERIFY_DIR,
    pdf: Callable[[Path, str], Document] = load_pdf_cached,
    html: Callable[[Path, str], Document] = load_html,
) -> Document:
    """One manifest entry as a `Document`, read by the format its entry records. Refuses a path not
    in the manifest, an entry not yet flagged, and a scan."""
    entry = read_manifest(verify_dir).get(name)
    if entry is None:
        raise KeyError(f"{name} is not in {verify_dir / MANIFEST_NAME}")
    if not {"format", "pages", "text_less_pages"} <= entry.keys():
        raise ValueError(f"{name} is not flagged yet; run flag_documents first")
    path = verify_dir / name
    if entry["format"] == "html":
        return html(path, path.stem)
    if is_scan(entry):
        raise ValueError(f"{name} is a scan: no text layer on pages {entry['text_less_pages']}")
    return pdf(path, path.stem)
