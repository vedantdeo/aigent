"""A filed PDF's tables re-rowed by where their printed lines sit: Docling keeps each table's box,
columns and header rows, and each body row is one printed line of the PDF's own text."""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from docling_core.types.doc.items.table.table import TableItem

from aigent.facts.cfg_facts import VERIFY_LINE_GAP
from aigent.facts.printed import number

Box = tuple[float, float, float, float]  # left, top, right, bottom on the page as displayed
PdfBox = tuple[float, float, float, float]  # left, bottom, right, top in the page's own space
PdfWord = tuple[str, PdfBox, bool]  # its text, box, and whether a space follows it
Word = tuple[str, Box, bool]
Rows = tuple[tuple[str, ...], ...]
Char = tuple[str, PdfBox | None]  # a character and its box; None for a space or line break

GRID = 8.0  # points per cell of the index that finds a run's characters
SEPARATOR = re.compile(r"(?<=\d)\s*([,.])\s*(?=\d)")  # 17 , 009 as an OCR layer prints it


def split_runs(chars: Sequence[Char], runs: Sequence[tuple[str, PdfBox]]) -> list[PdfWord]:
    """Each text run split at the spaces between its characters, a piece boxed by its own; a run
    none of whose characters lie inside it is kept whole. A run's last piece is followed by a
    space only when the run's text ends in one."""
    centres: dict[int, tuple[float, float]] = {
        i: ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        for i, (_, box) in enumerate(chars)
        if box is not None
    }
    grid: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in centres.items():
        grid.setdefault((int(x // GRID), int(y // GRID)), []).append(i)
    used: set[int] = set()
    out: list[PdfWord] = []
    for text, (left, bottom, right, top) in runs:
        spaced = text.endswith(" ")  # a line break alone does not end a word
        near = {
            i
            for gx in range(int((left - 1) // GRID), int((right + 1) // GRID) + 1)
            for gy in range(int((bottom - 1) // GRID), int((top + 1) // GRID) + 1)
            for i in grid.get((gx, gy), [])
        }
        mine = sorted(
            i
            for i in near - used
            if left - 0.5 <= centres[i][0] <= right + 0.5
            and bottom - 0.5 <= centres[i][1] <= top + 0.5
        )
        if not mine:
            out.append((text.strip(), (left, bottom, right, top), spaced))
            continue
        used.update(mine)
        word, box, last = "", None, -1
        for i in mine:
            if word and any(chars[j][0].isspace() for j in range(last + 1, i)):
                out.append((word, box or (left, bottom, right, top), True))
                word, box = "", None
            char, cbox = chars[i]
            if cbox is None:
                continue
            word += char
            box = cbox if box is None else _union(box, cbox)
            last = i
        if word:
            out.append((word, box or (left, bottom, right, top), spaced))
    return [w for w in out if w[0]]


def _union(a: PdfBox, b: PdfBox) -> PdfBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def page_words(page: pdfium.PdfPage) -> list[PdfWord]:
    """A PDF page's words in its own space, measured from its crop box's corner."""
    text = page.get_textpage()
    try:
        chars: list[Char] = []
        for i in range(text.count_chars()):
            char = chr(pdfium_c.FPDFText_GetUnicode(text.raw, i))
            chars.append((char, None if char.isspace() else text.get_charbox(i)))
        runs = [
            (text.get_text_bounded(*box), box)
            for box in (text.get_rect(i) for i in range(text.count_rects()))
        ]
        left, bottom, _, _ = crop_box(page)
        return [
            (word, (x1 - left, y1 - bottom, x2 - left, y2 - bottom), spaced)
            for word, (x1, y1, x2, y2), spaced in split_runs(chars, runs)
        ]
    finally:
        text.close()


def crop_box(page: pdfium.PdfPage) -> PdfBox:
    """The part of a page shown, as left, bottom, right, top; its media box when it has no crop."""
    box = page.get_cropbox() or page.get_mediabox()
    if box is None:
        raise ValueError("a PDF page with neither a crop box nor a media box")
    return box


def displayed(box: PdfBox, rotation: int, width: float, height: float) -> Box:
    """A box in the page's own space, `width` by `height` unrotated, on the page as displayed
    after turning it `rotation` degrees clockwise."""
    left, bottom, right, top = box
    corners = {
        0: ((left, height - top), (right, height - bottom)),
        90: ((bottom, left), (top, right)),
        180: ((width - right, bottom), (width - left, top)),
        270: ((height - top, width - right), (height - bottom, width - left)),
    }[rotation % 360]
    (x1, y1), (x2, y2) = corners
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def on_display(words: Sequence[PdfWord], rotation: int, width: float, height: float) -> list[Word]:
    """A page's words placed on the page as displayed (`displayed`)."""
    return [(word, displayed(box, rotation, width, height), spaced) for word, box, spaced in words]


@dataclass(frozen=True)
class Layout:
    """What Docling settled about a table: its box, how many of its rows are header, where its
    body begins, and each column's left and right edge."""

    box: Box
    header: int
    body_top: float
    columns: dict[int, tuple[float, float]]
    width: int


def layout(item: TableItem, rows: Rows, height: float) -> Layout | None:
    """A Docling table's layout; None for one split across pages, with no row of two figures, or
    with under two placed columns. Its header is every row above its first of two figures."""
    if len(item.prov) != 1:
        return None
    first = next(
        (i for i, row in enumerate(rows) if sum(number(c) is not None for c in row) >= 2), None
    )
    if first is None:
        return None
    tops: list[float] = []
    edges: dict[int, list[tuple[float, float]]] = {}
    for cell in item.data.table_cells:
        if cell.bbox is None or not cell.text.strip():
            continue
        box = cell.bbox.to_top_left_origin(height)
        if cell.start_row_offset_idx >= first:
            tops.append(box.t)
        if cell.col_span == 1:
            edges.setdefault(cell.start_col_offset_idx, []).append((box.l, box.r))
    if not tops or len(edges) < 2:
        return None
    columns = {
        j: (statistics.median(a for a, _ in spans), statistics.median(b for _, b in spans))
        for j, spans in edges.items()
    }
    box = item.prov[0].bbox.to_top_left_origin(height).as_tuple()
    return Layout(box, first, min(tops) - 1, columns, item.data.num_cols)


def line_rows(
    header: Rows, table: Layout, words: Sequence[Word], gap: float = VERIFY_LINE_GAP
) -> Rows | None:
    """`header` then one row per printed line of the table's body, each word in the column it
    overlaps most, else the nearest; a word with no space after it runs on into the next within
    a line's height. None when no word lies in the body."""
    left, _, right, bottom = table.box
    inside = sorted(
        (
            word
            for word in words
            if left - 2 <= (word[1][0] + word[1][2]) / 2 <= right + 2
            and table.body_top <= (word[1][1] + word[1][3]) / 2 <= bottom + 2
        ),
        key=lambda w: (w[1][1] + w[1][3]) / 2,
    )
    if not inside:
        return None
    height = statistics.median(box[3] - box[1] for _, box, _ in inside)
    lines: list[list[Word]] = []
    middle = 0.0
    for word in inside:
        y = (word[1][1] + word[1][3]) / 2
        if lines and y - middle <= height * gap:
            lines[-1].append(word)
            middle = statistics.fmean((b[1] + b[3]) / 2 for _, b, _ in lines[-1])
        else:
            lines.append([word])
            middle = y
    rows: list[tuple[str, ...]] = []
    for line in lines:
        cells = [""] * table.width
        for text, box in joined(sorted(line, key=lambda w: w[1][0]), height):
            j = column(box, table.columns)
            cells[j] = f"{cells[j]} {text}".strip()
        rows.append(tuple(SEPARATOR.sub(r"\1", cell) for cell in cells))
    return tuple(header) + tuple(rows)


def joined(line: Sequence[Word], height: float) -> list[tuple[str, Box]]:
    """A line's words left to right, each with no space after it joined to the next when that
    starts within `height` of its end."""
    out: list[tuple[str, Box]] = []
    runs_on = False
    for text, box, spaced in line:
        if runs_on and out and abs(box[0] - out[-1][1][2]) < height:
            before, (l0, t0, _, b0) = out[-1]
            out[-1] = (before + text, (l0, min(t0, box[1]), box[2], max(b0, box[3])))
        else:
            out.append((text, box))
        runs_on = not spaced
    return out


def column(box: Box, columns: dict[int, tuple[float, float]]) -> int:
    """The column a word overlaps most; for one overlapping none, the column nearest its middle."""
    overlap = {j: min(box[2], b) - max(box[0], a) for j, (a, b) in columns.items()}
    best = max(overlap, key=lambda j: overlap[j])
    if overlap[best] > 0:
        return best
    middle = (box[0] + box[2]) / 2
    return min(columns, key=lambda j: abs(middle - (columns[j][0] + columns[j][1]) / 2))
