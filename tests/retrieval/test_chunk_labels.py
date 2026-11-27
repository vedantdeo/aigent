"""A chunk's scale label checked by underhood: a scaled figure provable from its chunk."""

from __future__ import annotations

from cltrplusf.chunk import Document, by_element
from cltrplusf.corpus import assembled, elements_of
from cltrplusf.testing import Grid, Heading, Piece, docling
from underhood.tooldata.labels import Label, check_call


def _assembled(pages: list[list[Piece]]) -> Document:
    return assembled("RELIANCE-FY25", elements_of(docling(pages)))


def test_a_scale_in_a_label_lets_a_scaled_figure_be_checked_against_its_chunk() -> None:
    """The point of the label: a number whose scale the table's heading states is now provable."""
    rows = tuple(("Revenue from operations", "9,74,864", "9,01,064") for _ in range(1))
    table = Grid(rows=(("Particulars", "2025", "2024"), *rows), header_rows=1)
    document = _assembled([[Heading("Statement of Profit and Loss"), "(Rs in crore)", table]])

    [chunk] = [c for c in by_element()(document) if c.text.startswith("[Table")]

    label = Label("revenue", 9_748_640_000_000.0, "INR", 2025, None, "974864 * 10**7")
    assert check_call(label, chunk.text), chunk.text
