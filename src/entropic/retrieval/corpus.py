"""Annual report PDFs into the documents a chunker splits.

`pypdf` per page, each page normalised, then joined while recording the offset each page begins at,
so a citation still names the right page after extraction has reflowed the text. Running headers and
page numbers go by how often they repeat, since no markup survives a PDF to say what is furniture.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfReader

from entropic.config import (
    FURNITURE_EDGE_LINES,
    FURNITURE_MIN_PAGES,
    FURNITURE_RATIO,
    PAGE_SEPARATOR,
)
from entropic.retrieval.chunk import Document

REPO = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO / "corpus"
MANIFEST = REPO / "evals" / "reference" / "corpus-manifest.json"

_HYPHENATED_BREAK = re.compile(r"(\w)-\n(\w)")
_RUN_OF_SPACES = re.compile(r" {2,}")
_RUN_OF_BLANK_LINES = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


def read_pages(path: Path) -> list[str]:
    """Raw extracted text, one string per page. The only function here that touches `pypdf`.

    A page with no text layer comes back empty rather than missing, which is invariant 19 starting
    at the first function that could break it.
    """
    return [page.extract_text() or "" for page in PdfReader(path).pages]


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


def strip_furniture(
    pages: Sequence[str],
    *,
    min_pages: int = FURNITURE_MIN_PAGES,
    edge_lines: int = FURNITURE_EDGE_LINES,
    ratio: float = FURNITURE_RATIO,
) -> list[str]:
    """Drop the running headers, footers and page numbers a PDF repeats on every page.

    Only the first and last `edge_lines` of a page are candidates, so a label repeated mid-page is
    content. Candidates are counted with their digits masked, since a header carries the page
    number and so never repeats verbatim; one reaching `ratio` of the pages goes, wherever on a
    page it appears. Documents of fewer than `min_pages` pages come back untouched, and a page
    that was nothing but furniture stays, empty, or every citation after it shifts by one.
    """
    if len(pages) < min_pages:
        return list(pages)

    page_lines = [page.split("\n") for page in pages]
    seen: Counter[str] = Counter()
    for lines in page_lines:
        content = [line for line in lines if line.strip()]
        edges = content[:edge_lines] + content[-edge_lines:]
        seen.update({_skeleton(line) for line in edges})

    furniture = {
        line for line, pages_with_it in seen.items() if pages_with_it >= ratio * len(pages)
    }
    return [
        "\n".join(
            line for line in lines if not line.strip() or _skeleton(line) not in furniture
        ).strip()
        for lines in page_lines
    ]


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


def load_pdf(path: Path, doc_id: str | None = None) -> Document:
    """One PDF as a Document: read, normalise, strip furniture, join.

    `doc_id` defaults to the filename stem, which is what every chunk id is then built from.
    """
    pages = strip_furniture([normalise(page) for page in read_pages(path)])
    return to_document(doc_id or path.stem, pages, source=str(path))


def load_corpus(directory: Path = CORPUS_DIR) -> list[Document]:
    """Every PDF directly in a directory, in filename order, so two runs build the same chunk ids.

    A directory with no PDFs raises rather than returning nothing: an empty corpus scores zero
    recall on every question and looks like a bad retriever.
    """
    paths = sorted(directory.glob("*.pdf"))
    if not paths:
        raise ValueError(f"no PDFs in {directory} — run ./scripts/fetch-corpus.sh to download them")
    return [load_pdf(path) for path in paths]
