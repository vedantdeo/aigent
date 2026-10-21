"""Search keys over Docling documents built in code, and the keyed search over fake models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from aigent.retrieval import keys as module
from aigent.retrieval.chunk import Document, Inventory, Table, by_element
from aigent.retrieval.corpus import assembled, elements_of
from aigent.retrieval.hits import Hit
from aigent.retrieval.keys import (
    ParentRanker,
    build_keys,
    column_headers,
    keyed_search,
    row_label,
    table_context,
    years_in,
)
from aigent.retrieval.rank import Scope

from ..conftest import BagOfWordsEmbedder
from .conftest import Grid, Heading, Piece, docling

STATEMENT = Grid(
    rows=(
        ("(Rs in crore)", "", ""),
        ("Particulars", "2025", "2024"),
        ("Revenue from operations", "9,74,864", "9,01,064"),
        ("Other income", "17,978", "16,179"),
        ("-", "1", "2"),
        ("", "100", "90"),
    ),
    spans=((0, 0, 3, "(Rs in crore)"),),
    header_rows=2,
)
CONTEXT = "Consolidated Statement of Profit and Loss · consolidated"


def _document(pages: list[list[Piece]]) -> Document:
    return assembled("RELIANCE-FY25", elements_of(docling(pages)))


DOCUMENT = _document(
    [
        [Heading("Consolidated Statement of Profit and Loss"), STATEMENT],
        [
            Heading("Risk management"),
            "Risks are reviewed each quarter by the Board. The Board met four times.",
        ],
        [Heading("Auditors"), "The auditors were reappointed."],
    ]
)


def _keys(document: Document = DOCUMENT, max_chars: int = 2000) -> tuple[Inventory, module.Keyed]:
    chunks = Inventory.build("t", [document], by_element(max_chars=max_chars))
    return chunks, build_keys([document], chunks)


def _of(keyed: module.Keyed, kind: str) -> list[tuple[str, str]]:
    return [(k.text, keyed.parent[k.id]) for k in keyed.keys.chunks if keyed.kind[k.id] == kind]


def test_every_chunk_is_a_key_of_itself() -> None:
    chunks, keyed = _keys()

    assert _of(keyed, "chunk") == [(c.text, c.id) for c in chunks.chunks]
    assert keyed.keys.name == "t-keys", "its own name, so its cached vectors are its own"
    assert len({k.id for k in keyed.keys.chunks}) == len(keyed.keys), "key ids are unique"


def test_a_table_gets_a_number_free_key_per_labelled_row_and_one_listing_them() -> None:
    chunks, keyed = _keys()
    [table] = [c.id for c in chunks.chunks if c.text.startswith("[Table")]

    assert _of(keyed, "row") == [
        (f"{CONTEXT} · Revenue from operations · 2025, 2024", table),
        (f"{CONTEXT} · Other income · 2025, 2024", table),
    ], "no key for a placeholder label or a row of figures"
    assert _of(keyed, "table") == [
        (f"{CONTEXT} · 2025, 2024\nrows: Revenue from operations; Other income", table)
    ]


def test_a_row_key_points_at_the_piece_of_a_split_table_that_prints_it() -> None:
    rows = tuple((f"Line item {name}", f"{n},000", f"{n},500") for n, name in enumerate("abcdefgh"))
    grid = Grid(rows=(("Particulars", "2025", "2024"), *rows), header_rows=1)
    chunks, keyed = _keys(_document([[Heading("Segment results"), grid]]), max_chars=160)
    pieces = [c for c in chunks.chunks if c.text.startswith("[Table")]

    assert len(pieces) > 1, "the fixture must split"
    for text, parent in _of(keyed, "row"):
        label = text.split(" · ")[-2]
        assert label in chunks.by_id[parent].text, (label, parent)
    tables = _of(keyed, "table")
    assert [parent for _, parent in tables] == [p.id for p in pieces], "one table key per piece"


def test_a_text_chunk_gets_a_key_per_sentence_under_its_heading_without_repeating_it() -> None:
    chunks, keyed = _keys()
    risk = next(c.id for c in chunks.chunks if c.heading == "Risk management")

    assert _of(keyed, "sentence") == [
        ("Risk management · Risks are reviewed each quarter by the Board.", risk),
        ("Risk management · The Board met four times.", risk),
    ], "a chunk of one sentence (Auditors) is already its own key"


@pytest.mark.parametrize(
    ("label", "context"),
    [
        pytest.param(
            "[Table · ITC-FY25 p.251 · Consolidated Statement of Profit and Loss · consolidated"
            " · (₹ in Crores) = ×10000000 · columns: 2025 | 2024]",
            "Consolidated Statement of Profit and Loss · consolidated",
            id="a full label keeps its section and basis",
        ),
        pytest.param(
            "[Table · RELIANCE-FY25 p.12 · FINANCIAL PERFORMANCE · (₹ Crore) = ×10000000]",
            "FINANCIAL PERFORMANCE",
            id="a label with no basis",
        ),
        pytest.param(
            "[Table · TATAMOTORS-FY25 p.258 · D. Cash Flow · ₹ in crores · columns: 2025]",
            "D. Cash Flow",
            id="a scale with no multiplier is dropped too",
        ),
        pytest.param("[Table · ITC-FY25 p.4]", "", id="a label with nothing but its page"),
    ],
)
def test_a_tables_context_is_its_section_and_basis_alone(label: str, context: str) -> None:
    assert table_context(label) == context


@pytest.mark.parametrize(
    ("row", "label"),
    [
        (("Revenue from operations", "25", "9,80,136"), "Revenue from operations"),
        (("", "Finance Costs", "24,269"), "Finance Costs"),
        (("3A.", "Property, Plant and Equipment", "1"), "Property, Plant and Equipment"),
        pytest.param(("-", "1", "2"), None, id="a dash is a placeholder"),
        pytest.param(("(e)", "1"), None, id="a letter in brackets is a placeholder"),
        pytest.param(("", "100", "90"), None, id="a row of figures has no label"),
    ],
)
def test_a_rows_label_is_its_first_cell_of_words(row: tuple[str, ...], label: str | None) -> None:
    assert row_label(row) == label


@pytest.mark.parametrize(
    ("header_rows", "headers", "years"),
    [
        pytest.param(
            (("", "Notes", "2024-25", "(₹ in crore) 2023-24"),),
            ["", "Notes", "2024-25", "(₹ in crore) 2023-24"],
            ["2024-25", "2023-24"],
            id="one header row, a scale beside a year",
        ),
        pytest.param(
            (("", "FY25", "FY25", "FY24"), ("", "Wholesale", "Retail", "Wholesale")),
            ["", "FY25 Wholesale", "FY25 Retail", "FY24 Wholesale"],
            ["FY25", "FY24"],
            id="two header rows read top to bottom, a year once",
        ),
        pytest.param((("Director", "Category"),), ["Director", "Category"], [], id="no years"),
    ],
)
def test_column_headers_and_their_years(
    header_rows: tuple[tuple[str, ...], ...], headers: list[str], years: list[str]
) -> None:
    table = Table(0, 1, (*header_rows, ("Revenue", "1", "2", "3")), header_rows=len(header_rows))

    assert column_headers(table)[: len(headers)] == headers
    assert years_in(column_headers(table)) == years


@dataclass
class _Keys:
    """Ranks four keys of three chunks, recording what it was asked."""

    asked: list[tuple[int, Scope]] = field(default_factory=list[tuple[int, Scope]])

    def rank(self, query: str, k: int = 5, *, doc_id: Scope = None) -> list[Hit]:
        self.asked.append((k, doc_id))
        return [Hit("k1", 0.9), Hit("k2", 0.8), Hit("k3", 0.7), Hit("k4", 0.6)][:k]


PARENT = {"k1": "A", "k2": "A", "k3": "B", "k4": "C"}


@pytest.mark.parametrize(
    ("k", "found"),
    [
        pytest.param(2, [Hit("A", 0.9), Hit("B", 0.7)], id="each chunk once, at its best key"),
        pytest.param(5, [Hit("A", 0.9), Hit("B", 0.7), Hit("C", 0.6)], id="fewer chunks than k"),
    ],
)
def test_a_parent_ranker_returns_chunks_not_keys(k: int, found: list[Hit]) -> None:
    keys = _Keys()

    assert ParentRanker(keys, PARENT, depth=40).rank("q", k, doc_id="ITC-FY25") == found
    assert keys.asked == [(40, "ITC-FY25")], "reads its depth of keys, scoped as asked"


@dataclass
class _Reranker:
    """Scores passages by the word 'income', recording what it was handed."""

    name: str = "income-fake"
    read: list[list[str]] = field(default_factory=list[list[str]])

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        self.read.append(list(passages))
        return [float(p.lower().count("income")) for p in passages]


@pytest.mark.parametrize(
    ("method", "reranked"),
    [("hybrid+rerank", True), ("hybrid", False)],
    ids=["with a reranker", "without one"],
)
def test_a_keyed_search_returns_whole_chunks(
    monkeypatch: pytest.MonkeyPatch, method: str, reranked: bool
) -> None:
    monkeypatch.setattr(module, "SEARCH_METHOD", method)
    reranker = _Reranker()

    chunks, ranker = keyed_search(
        "filings", [DOCUMENT], cache=False, embedder=BagOfWordsEmbedder(), reranker=reranker
    )
    hits = ranker.rank("other income", 2, doc_id="RELIANCE-FY25")

    assert chunks.name == "filings"
    assert hits and all(hit.chunk_id in chunks.by_id for hit in hits), "chunk ids, never key ids"
    assert chunks.by_id[hits[0].chunk_id].text.startswith("[Table"), hits
    if reranked:
        assert reranker.read and all(p in chunks.texts() for p in reranker.read[0]), (
            "the reranker reads whole chunks, not keys"
        )
    else:
        assert reranker.read == []
