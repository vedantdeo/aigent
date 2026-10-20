"""Docling documents built in code, and a fake converter, so no test downloads Docling's models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table_data import TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel
from pypdf import PdfReader

from aigent.retrieval import corpus


@dataclass(frozen=True)
class Heading:
    text: str


@dataclass(frozen=True)
class Footer:
    text: str


@dataclass(frozen=True)
class Grid:
    """A table: its rows of cells, cells spanning columns as (row, first, last, text), and how many
    leading rows are column headers."""

    rows: tuple[tuple[str, ...], ...]
    spans: tuple[tuple[int, int, int, str], ...] = ()
    header_rows: int = 0


@dataclass(frozen=True)
class Picture:
    """A picture with these lines of text drawn inside it."""

    texts: tuple[str, ...]


Piece = str | Heading | Footer | Grid | Picture


def _prov(page: int, text: str) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(text))
    )


def _table(grid: Grid) -> TableData:
    cells: list[TableCell] = []
    spanned = {(row, column) for row, first, last, _ in grid.spans for column in range(first, last)}
    for row, first, last, text in grid.spans:
        cells.append(
            TableCell(
                text=text,
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=first,
                end_col_offset_idx=last,
                col_span=last - first,
                column_header=row < grid.header_rows,
            )
        )
    for r, values in enumerate(grid.rows):
        for c, text in enumerate(values):
            if (r, c) not in spanned:
                cells.append(
                    TableCell(
                        text=text,
                        start_row_offset_idx=r,
                        end_row_offset_idx=r + 1,
                        start_col_offset_idx=c,
                        end_col_offset_idx=c + 1,
                        column_header=r < grid.header_rows,
                    )
                )
    columns = max(len(values) for values in grid.rows)
    return TableData(table_cells=cells, num_rows=len(grid.rows), num_cols=columns)


def docling(pages: Sequence[Sequence[Piece]]) -> DoclingDocument:
    """A Docling document with these pieces on these pages, in this order."""
    document = DoclingDocument(name="report")
    for number, pieces in enumerate(pages, start=1):
        document.add_page(page_no=number, size=Size(width=612, height=792))
        for piece in pieces:
            if isinstance(piece, Heading):
                document.add_heading(text=piece.text, prov=_prov(number, piece.text))
            elif isinstance(piece, Footer):
                document.add_text(
                    label=DocItemLabel.PAGE_FOOTER,
                    text=piece.text,
                    prov=_prov(number, piece.text),
                    content_layer=ContentLayer.FURNITURE,
                )
            elif isinstance(piece, Grid):
                document.add_table(data=_table(piece), prov=_prov(number, "table"))
            elif isinstance(piece, Picture):
                picture = document.add_picture(prov=_prov(number, "picture"))
                for text in piece.texts:
                    document.add_text(
                        label=DocItemLabel.TEXT, text=text, prov=_prov(number, text), parent=picture
                    )
            else:
                document.add_text(label=DocItemLabel.TEXT, text=piece, prov=_prov(number, piece))
    return document


def lines_of(path: Path) -> DoclingDocument:
    """What the fake converter reads a test PDF as: each printed line a text item on its page."""
    pages = [(page.extract_text() or "").split("\n") for page in PdfReader(path).pages]
    return docling([[line for line in lines if line.strip()] for lines in pages])


@pytest.fixture
def fake_docling(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """`corpus.convert` reading a test PDF line by line; the paths it was asked to convert."""
    converted: list[Path] = []

    def convert(path: Path) -> DoclingDocument:
        converted.append(path)
        return lines_of(path)

    monkeypatch.setattr(corpus, "convert", convert)
    return converted
