"""Docling documents built in code, and a fake converter, so no test downloads Docling's models."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
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
class Running:
    """A running page header, set aside by Docling as furniture."""

    text: str


@dataclass(frozen=True)
class Grid:
    """A table: its rows of cells, cells spanning columns as (row, first, last, text), and how many
    leading rows are column headers."""

    rows: tuple[tuple[str, ...], ...]
    spans: tuple[tuple[int, int, int, str], ...] = ()
    header_rows: int = 0
    # Laid out on the page when given: (left, bottom, right, top) from the bottom-left corner, cells
    # tiling it below a caption band of `band` points and left of an empty `margin` on the right.
    box: tuple[float, float, float, float] | None = None
    band: float = 0.0
    margin: float = 0.0
    caption: str | None = None  # a caption Docling found itself
    footnote: str | None = None
    sections: tuple[int, ...] = ()  # rows Docling marks as opening a section


@dataclass(frozen=True)
class Picture:
    """A picture with these lines of text drawn inside it."""

    texts: tuple[str, ...]


Piece = str | Heading | Footer | Running | Grid | Picture


def _prov(page: int, text: str) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(text))
    )


def _cell_box(grid: Grid, row: int, first: int, last: int) -> BoundingBox | None:
    if grid.box is None:
        return None
    left, bottom, right, top = grid.box
    width = (right - left - grid.margin) / max(len(values) for values in grid.rows)
    height = (top - bottom - grid.band) / len(grid.rows)
    upper = top - grid.band - row * height
    return BoundingBox(
        l=left + first * width,
        r=left + last * width,
        t=upper,
        b=upper - height,
        coord_origin=CoordOrigin.BOTTOMLEFT,
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
                bbox=_cell_box(grid, row, first, last),
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
                        row_section=r in grid.sections and c == 0,
                        bbox=_cell_box(grid, r, c, c + 1),
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
            elif isinstance(piece, Running):
                document.add_text(
                    label=DocItemLabel.PAGE_HEADER,
                    text=piece.text,
                    prov=_prov(number, piece.text),
                    content_layer=ContentLayer.FURNITURE,
                )
            elif isinstance(piece, Footer):
                document.add_text(
                    label=DocItemLabel.PAGE_FOOTER,
                    text=piece.text,
                    prov=_prov(number, piece.text),
                    content_layer=ContentLayer.FURNITURE,
                )
            elif isinstance(piece, Grid):
                prov = _prov(number, "table")
                if piece.box is not None:
                    left, bottom, right, top = piece.box
                    prov.bbox = BoundingBox(
                        l=left, b=bottom, r=right, t=top, coord_origin=CoordOrigin.BOTTOMLEFT
                    )
                table = document.add_table(data=_table(piece), prov=prov)
                if piece.caption is not None:
                    caption = document.add_text(
                        label=DocItemLabel.CAPTION,
                        text=piece.caption,
                        prov=_prov(number, piece.caption),
                        parent=table,
                    )
                    table.captions.append(caption.get_ref())
                if piece.footnote is not None:
                    footnote = document.add_text(
                        label=DocItemLabel.FOOTNOTE,
                        text=piece.footnote,
                        prov=_prov(number, piece.footnote),
                        parent=table,
                    )
                    table.footnotes.append(footnote.get_ref())
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


def build_pdf(
    pages: list[list[str]],
    second_font: str = "Courier",
    encoding: bytes = b"/WinAnsiEncoding",
    to_unicode: dict[str, str] | None = None,
    boxed: str | None = None,
    loops: bool = False,
) -> bytes:
    """An uncompressed PDF with these lines on these pages, so a test can read one back. A part of
    a line in braces is drawn in `second_font`, with this encoding and, given one, a ToUnicode map
    from each character drawn to the text it reads as. `boxed` is a line every page draws from a
    form XObject, its braced part in a font object of its own of `second_font`'s family; with
    `loops`, the form lists itself among its own XObjects."""
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

    def drawn(lines: list[str]) -> bytes:
        text = b"BT /F1 12 Tf 72 720 Td 14 TL\n"
        for line in lines:
            for number, part in enumerate(re.split(r"[{}]", line)):
                escaped = part.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
                text += b"/F2 12 Tf " if number % 2 else b"/F1 12 Tf "
                text += b"(" + escaped.encode("cp1252") + b") Tj\n"
            text += b"T*\n"
        return text + b"ET"

    xobject = b""
    if boxed is not None:
        font3 = add(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /"
            + second_font.encode()
            + b" /Encoding "
            + encoding
            + b" >>"
        )
        form = drawn([boxed])
        box = len(objects) + 1
        itself = b" /XObject << /X1 " + str(box).encode() + b" 0 R >>" if loops else b""
        add(
            b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 "
            + str(font).encode()
            + b" 0 R /F2 "
            + str(font3).encode()
            + b" 0 R >>"
            + itself
            + b" >> /Length "
            + str(len(form)).encode()
            + b" >>\nstream\n"
            + form
            + b"\nendstream"
        )
        xobject = b" /XObject << /X1 " + str(box).encode() + b" 0 R >>"

    kids: list[int] = []
    for lines in pages:
        text = drawn(lines) + (b"\nq 1 0 0 1 0 -400 cm /X1 Do Q" if boxed is not None else b"")
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
                + b" 0 R >>"
                + xobject
                + b" >> >>"
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
