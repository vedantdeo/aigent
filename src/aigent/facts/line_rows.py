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

from aigent.facts.cfg_facts import VERIFY_BAND_MIN_SHARE, VERIFY_LINE_GAP, VERIFY_WRAP_GAP
from aigent.facts.printed import ENUMERATOR, figure_row, number

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
    body begins, each column's left and right edge, each header cell's row, column, span and box,
    and the columns whose body cells are mostly words."""

    box: Box
    header: int
    body_top: float
    columns: dict[int, tuple[float, float]]
    width: int
    cells: tuple[tuple[int, int, int, Box], ...] = ()
    text: frozenset[int] = frozenset()


def layout(item: TableItem, rows: Rows, height: float) -> Layout | None:
    """A Docling table's layout; None for one split across pages, with no row of two figures, or
    with under two placed columns. Its header is every row above its first of two figures."""
    if len(item.prov) != 1:
        return None
    first = next((i for i, row in enumerate(rows) if figure_row(row)), None)
    if first is None:
        return None
    tops: list[float] = []
    edges: dict[int, list[tuple[float, float]]] = {}
    cells: list[tuple[int, int, int, Box]] = []
    for cell in item.data.table_cells:
        if cell.bbox is None or not cell.text.strip():
            continue
        box = cell.bbox.to_top_left_origin(height)
        if cell.start_row_offset_idx >= first:
            tops.append(box.t)
        elif cell.start_row_offset_idx >= 0 and cell.start_col_offset_idx >= 0:
            span = cell.end_col_offset_idx - cell.start_col_offset_idx
            cells.append(
                (cell.start_row_offset_idx, cell.start_col_offset_idx, span, box.as_tuple())
            )
        if cell.col_span == 1:
            edges.setdefault(cell.start_col_offset_idx, []).append((box.l, box.r))
    if not tops or len(edges) < 2:
        return None
    columns = {
        j: (statistics.median(a for a, _ in spans), statistics.median(b for _, b in spans))
        for j, spans in edges.items()
    }
    box = item.prov[0].bbox.to_top_left_origin(height).as_tuple()
    filled: dict[int, list[str]] = {}
    for row in rows[first:]:
        for j, c in enumerate(row):
            if c:
                filled.setdefault(j, []).append(c)
    text = frozenset(j for j, cs in filled.items() if sum(map(_printed, cs)) * 2 < len(cs))
    return Layout(
        box, first, min(tops) - 1, columns, item.data.num_cols, tuple(sorted(cells)), text
    )


def _middle(box: Box) -> float:
    return (box[1] + box[3]) / 2


def _lines(words: Sequence[Word], gap: float) -> tuple[list[list[Word]], float]:
    """Words grouped into printed lines top to bottom, and their median height."""
    ordered = sorted(words, key=_middle_of)
    height = statistics.median(box[3] - box[1] for _, box, _ in ordered)
    lines: list[list[Word]] = []
    middle = 0.0
    for word in ordered:
        y = _middle(word[1])
        if lines and y - middle <= height * gap:
            lines[-1].append(word)
            middle = statistics.fmean(_middle(b) for _, b, _ in lines[-1])
        else:
            lines.append([word])
            middle = y
    return lines, height


def _middle_of(word: Word) -> float:
    return _middle(word[1])


_FIGURE = re.compile(r"[(\[]?[-+]?[\d,.]*\d[\d,.]*%?[)\]]?%?")  # 1,234.5  (12)  (9.8%)


def _upright(words: Sequence[Word]) -> bool:
    """Whether most words of three or more characters are taller than wide: text run up or down
    the page as displayed."""
    long = [box for text, box, _ in words if len(text) >= 3]
    return sum(b[3] - b[1] > b[2] - b[0] for b in long) * 2 > len(long)


def _is_figure(text: str) -> bool:
    return bool(_FIGURE.fullmatch(text))


def _printed(text: str) -> bool:
    """A cell printing figures, or a dash for none, only."""
    tokens = text.split()
    return bool(tokens) and all(_is_figure(t) or number(t) is not None for t in tokens)


def bands(
    lines: Sequence[Sequence[tuple[str, Box]]],
    height: float,
    min_share: float,
    text: Sequence[tuple[float, float]] = (),
) -> list[tuple[float, float]]:
    """A table body's figure columns as printed, left to right: its figures clustered by right
    edge, neighbours never sharing a line joined, and one on under `min_share` of the lines with
    figures dropped, as is one centred in a `text` column or on fewer lines of two that overlap;
    each spans its figures."""
    figures = sorted(
        ((i, box) for i, line in enumerate(lines) for word, box in line if _printed(word)),
        key=lambda f: f[1][2],
    )
    clusters: list[list[tuple[int, Box]]] = []
    for figure in figures:
        if clusters and figure[1][2] - clusters[-1][-1][1][2] <= height:
            clusters[-1].append(figure)
        else:
            clusters.append([figure])
    joined_up: list[list[tuple[int, Box]]] = []
    for cluster in clusters:
        if joined_up:
            last = joined_up[-1]
            apart = {i for i, _ in last}.isdisjoint(i for i, _ in cluster)
            mine, theirs = _extent(cluster), _extent(last)
            if apart and mine[0] < theirs[1] and theirs[0] < mine[1]:
                last.extend(cluster)
                continue
        joined_up.append(cluster)
    least = max(2, min_share * len({i for i, _ in figures}))
    kept = sorted(
        (_extent(c), len({i for i, _ in c}))
        for c in joined_up
        if len({i for i, _ in c}) >= least
        and not any(a <= sum(_extent(c)) / 2 <= b for a, b in text)
    )
    while (
        clash := next((k for k in range(len(kept) - 1) if kept[k][0][1] > kept[k + 1][0][0]), None)
    ) is not None:
        del kept[clash if kept[clash][1] < kept[clash + 1][1] else clash + 1]
    return [span for span, _ in kept]


def _extent(figures: Sequence[tuple[int, Box]]) -> tuple[float, float]:
    return min(b[0] for _, b in figures), max(b[2] for _, b in figures)


def _apart(box: Box, span: tuple[float, float]) -> float:
    """How far a box lies from a column's edges; less than zero by as much as they overlap."""
    return max(span[0], box[0]) - min(span[1], box[2])


@dataclass(frozen=True)
class Grid:
    """A table's printed columns left to right: its figure columns (`bands`) and each of Docling's
    columns holding none, as a text column, by the Docling column it was."""

    spans: tuple[tuple[float, float], ...]
    figures: tuple[int, ...]
    text: dict[int, int]

    @property
    def width(self) -> int:
        return len(self.spans)

    def place(self, text: str, box: Box) -> int:
        """The column a piece of text prints in: the figure column it overlaps, a figure, or holds
        the middle of; a word left of every figure its nearest text column there; else the
        nearest column."""
        middle = (box[0] + box[2]) / 2
        bands = [(_apart(box, self.spans[k]), k) for k in self.figures]
        if _is_figure(text) and (best := min(bands))[0] < 0:
            return best[1]
        inside = [k for k in self.figures if self.spans[k][0] <= middle <= self.spans[k][1]]
        if inside:
            return inside[0]
        left = self.labels()
        if middle < self.spans[self.figures[0]][0] and left:
            return min(left, key=lambda k: _apart(box, self.spans[k]))
        return min(range(len(self.spans)), key=lambda k: _apart(box, self.spans[k]))

    def labels(self) -> list[int]:
        """The text columns left of every figure column."""
        first = self.spans[self.figures[0]][0]
        return [k for k in self.text.values() if self.spans[k][1] <= first + 2]

    def covered(self, left: float, right: float) -> list[int]:
        """The figure columns whose middles lie between `left` and `right`."""
        return [
            k
            for k in self.figures
            if left - 2 <= (self.spans[k][0] + self.spans[k][1]) / 2 <= right + 2
        ]

    def nearest_figures(self, box: Box) -> int:
        middle = (box[0] + box[2]) / 2
        return min(self.figures, key=lambda k: abs(middle - sum(self.spans[k]) / 2))


def grid_of(table: Layout, figures: list[tuple[float, float]]) -> Grid:
    """The printed columns of a table with these figure columns and Docling's columns of words;
    one text column across its left when Docling placed none there."""
    middles = [(a + b) / 2 for a, b in figures]
    text = {
        j: e
        for j, e in table.columns.items()
        if j in table.text and not any(e[0] <= m <= e[1] for m in middles)
    }
    if not any(b <= figures[0][0] + 2 for _, b in text.values()):
        text[-1] = (table.box[0], figures[0][0])
    placed = sorted(
        [(e, "text", j) for j, e in text.items()]
        + [(e, "figure", k) for k, e in enumerate(figures)]
    )
    return Grid(
        tuple(e for e, _, _ in placed),
        tuple(i for i, (_, kind, _) in enumerate(placed) if kind == "figure"),
        {j: i for i, (_, kind, j) in enumerate(placed) if kind == "text"},
    )


def header_rows(header: Rows, table: Layout, grid: Grid, words: Sequence[Word]) -> Rows:
    """Docling's header rows laid over the printed columns: a text column's cells kept in it, a
    cell over one figure column, or spanning several, given to its first; one Docling merged
    over several split into its stacks of phrases (`stacks`), one over a column given to it and
    one over several placed by `grouped`. A cell Docling
    gave no box is placed by its column."""
    out = [[""] * grid.width for _ in header]

    def put(i: int, k: int, text: str) -> None:
        out[i][k] = f"{out[i][k]} {text}".strip()

    boxed = {(i, j): (span, box) for i, j, span, box in table.cells}
    groups: dict[int, list[tuple[str, list[int]]]] = {}
    for i, row in enumerate(header):
        for j, text in enumerate(row):
            if text and (i, j) not in boxed:
                a, b = table.columns.get(j, grid.spans[grid.figures[0]])
                boxed[i, j] = (1, (a, 0.0, b, 0.0))  # placed by its column alone
    for (i, j), (span, box) in sorted(boxed.items()):
        text = header[i][j] if i < len(header) and j < len(header[i]) else ""
        if not text:
            continue
        if j in grid.text:
            put(i, grid.text[j], text)
            continue
        own = grid.covered(box[0], box[2])
        mine = [
            w
            for w in words
            if box[0] - 1 <= (w[1][0] + w[1][2]) / 2 <= box[2] + 1
            and box[1] - 1 <= _middle(w[1]) <= box[3] + 1
        ]
        if len(own) < 2 or span > 1 or not mine:
            theirs = grid.covered(*table.columns[j]) if j in table.columns else []
            k = theirs[0] if len(theirs) == 1 else own[0] if own else grid.nearest_figures(box)
            put(i, k, text)
            continue
        lines, height = _lines(mine, VERIFY_LINE_GAP)
        found = [
            phrase
            for line in lines
            for phrase in phrases(joined(sorted(line, key=lambda w: w[1][0]), height), height)
        ]
        for heading, at in stacks(found):
            over = grid.covered(at[0], at[2])
            if len(over) >= 2:
                groups.setdefault(i, []).append((heading, over))
            else:
                near = min(own, key=lambda k: _apart(at, grid.spans[k]))
                put(i, over[0] if over else near, heading)
    for i, found_ in groups.items():
        free = [k for k in grid.figures if not out[i][k]]  # those no heading of their own names
        for heading, k in grouped(found_, free):
            put(i, k, heading)
    return tuple(tuple(row) for row in out)


def grouped(
    headings: Sequence[tuple[str, list[int]]], figures: Sequence[int]
) -> list[tuple[str, int]]:
    """Group headings, each with the figure columns it is printed over, given the column each
    heads: the first of an equal block of the figure columns when each lies within its own
    block, else the first column it is printed over."""
    size, extra = divmod(len(figures), len(headings))
    blocks = [list(figures[n * size : (n + 1) * size]) for n in range(len(headings))]
    if not extra and all(
        set(over) <= set(b) for (_, over), b in zip(headings, blocks, strict=True)
    ):
        return [(heading, b[0]) for (heading, _), b in zip(headings, blocks, strict=True)]
    return [(heading, over[0]) for heading, over in headings]


_JOINS = re.compile(r"(?:\b(?:of|and|the|in|on|or|to|for|from|by|a|an|with|at)|[,/&-])$")
_OPENS = re.compile(r"(?:less|add|of which)\b")  # a row of its own, though in lower case


def _continues(label: str) -> bool:
    """A label starting in lower case that is not an enumerator, as `receivables considered`."""
    return label[:1].islower() and not ENUMERATOR.match(label) and not _OPENS.match(label)


def wrapped(
    lines: Sequence[tuple[float, tuple[str, ...]]], labels: frozenset[int], pitch: float
) -> Rows:
    """Printed lines as rows, each label wrapped over lines within `pitch` of each other joined
    into one row: one ending in a joining word or starting in lower case, or a pair of label lines
    with the row's figures printed between them."""

    def label(cells: Sequence[str]) -> str:
        return " ".join(c for j, c in enumerate(cells) if c and j in labels)

    def figures(cells: Sequence[str]) -> bool:
        return any(c for j, c in enumerate(cells) if j not in labels)

    def merge(*parts: Sequence[str]) -> list[str]:
        return [" ".join(p[j] for p in parts if p[j]) for j in range(len(parts[0]))]

    out: list[tuple[float, list[str]]] = []
    k = 0
    while k < len(lines):
        y, cells = lines[k][0], list(lines[k][1])
        if out and k + 1 < len(lines) and figures(cells) and not label(cells):
            (above, prev), (below, nxt) = out[-1], lines[k + 1]
            if (
                label(prev)
                and not figures(prev)
                and label(nxt)
                and not figures(nxt)
                and above < y < below
                and below - above <= pitch
            ):
                out[-1] = (below, merge(prev, cells, nxt))
                k += 2
                continue
        if out and y - out[-1][0] <= pitch and label(cells):
            above, prev = out[-1]
            if (
                label(prev)
                and not figures(prev)
                and (_JOINS.search(label(prev)) or _continues(label(cells)))
            ):
                out[-1] = (y, merge(prev, cells))
                k += 1
                continue
            if (
                figures(prev)
                and not figures(cells)
                and (_continues(label(cells)) or _JOINS.search(label(prev)))
            ):
                out[-1] = (y, merge(prev, cells))
                k += 1
                continue
        out.append((y, cells))
        k += 1
    return tuple(tuple(cells) for _, cells in out)


def line_rows(
    header: Rows,
    table: Layout,
    words: Sequence[Word],
    gap: float = VERIFY_LINE_GAP,
    min_share: float = VERIFY_BAND_MIN_SHARE,
    wrap: float = VERIFY_WRAP_GAP,
) -> Rows | None:
    """`header` then one row per printed line of the table's body. The columns are the figure
    columns its figures print in (`bands`) among Docling's text columns (`grid_of`), the header over
    them (`header_rows`) and wrapped labels joined (`wrapped`); with no figure column found,
    Docling's columns and header, each word in the column it overlaps most. None when no word
    lies in the body."""
    left, top, right, bottom = table.box
    within = [w for w in words if left - 2 <= (w[1][0] + w[1][2]) / 2 <= right + 2]
    inside = [w for w in within if table.body_top <= _middle(w[1]) <= bottom + 2]
    if not inside:
        return None
    lines, height = _lines(inside, gap)
    pieces = [split_figures(joined(sorted(line, key=lambda w: w[1][0]), height)) for line in lines]
    text = [table.columns[j] for j in table.text if j in table.columns]
    figures = [] if _upright(inside) else bands(pieces, height, min_share, text)
    if not figures:
        rows: list[tuple[str, ...]] = []
        for line in pieces:
            cells = [""] * table.width
            for text, box in line:
                j = column(box, table.columns)
                cells[j] = f"{cells[j]} {text}".strip()
            rows.append(tuple(SEPARATOR.sub(r"\1", cell) for cell in cells))
        return tuple(header) + tuple(rows)
    grid = grid_of(table, figures)
    placed: list[tuple[float, tuple[str, ...]]] = []
    for line, own in zip(pieces, lines, strict=True):
        cells = [""] * grid.width
        for text, box in line:
            k = grid.place(text, box)
            cells[k] = f"{cells[k]} {text}".strip()
        y = statistics.fmean(_middle(b) for _, b, _ in own)
        placed.append((y, tuple(SEPARATOR.sub(r"\1", cell) for cell in cells)))
    above = [w for w in within if top - 2 <= _middle(w[1]) < table.body_top]
    read = header_rows(header, table, grid, above) + wrapped(
        placed, frozenset(grid.labels()), height * wrap
    )
    kept = [k for k in range(grid.width) if any(row[k] for row in read)]
    return tuple(tuple(row[k] for k in kept) for row in read)


def phrases(line: Sequence[tuple[str, Box]], height: float) -> list[tuple[str, Box]]:
    """A line's pieces joined into phrases: each starting within `height` of the last's end."""
    out: list[tuple[str, Box]] = []
    for text, box in line:
        if out and box[0] - out[-1][1][2] <= height:
            before, (l0, t0, _, b0) = out[-1]
            out[-1] = (f"{before} {text}", (l0, min(t0, box[1]), box[2], max(b0, box[3])))
        else:
            out.append((text, box))
    return out


def stacks(found: Sequence[tuple[str, Box]]) -> list[tuple[str, Box]]:
    """Phrases, top to bottom, stacked where they overlap across the page: one heading printed
    over several lines, each read in order, boxed by them all."""
    out: list[tuple[list[str], Box]] = []
    for text, box in found:
        hits = [k for k, (_, b) in enumerate(out) if box[0] < b[2] and b[0] < box[2]]
        if not hits:
            out.append(([text], box))
            continue
        texts: list[str] = []
        joined_box = box
        for k in hits:
            texts += out[k][0]
            joined_box = _union(joined_box, out[k][1])
        for k in reversed(hits):
            del out[k]
        out.append(([*texts, text], joined_box))
    return sorted(((" ".join(t), b) for t, b in out), key=lambda s: s[1][0])


def split_figures(line: Sequence[tuple[str, Box]]) -> list[tuple[str, Box]]:
    """A line's pieces with one printing several figures apart, as an OCR layer's run can, cut
    into one piece a figure, each boxed by where its characters fall along the run."""
    out: list[tuple[str, Box]] = []
    for text, box in line:
        tokens = text.split()
        if len(tokens) < 2 or not all(_printed(t) for t in tokens):
            out.append((text, box))
            continue
        step = (box[2] - box[0]) / len(text)
        start = 0
        for token in tokens:
            at = text.index(token, start)
            start = at + len(token)
            out.append((token, (box[0] + at * step, box[1], box[0] + start * step, box[3])))
    return out


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
