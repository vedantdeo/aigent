"""Search keys: every chunk indexed under several short texts that each point back to it.

A table chunk is also found by one key per row (its section, basis, row label and column years) and
one key listing its rows; a text chunk by one key per sentence; every chunk by its own text. Keys
carry no figures. `keyed_search` ranks the keys and hands back the chunks they stand for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from aigent.config import KEY_DEPTH, RERANK_CANDIDATES, SEARCH_METHOD, TOP_K
from aigent.retrieval.chunk import (
    Chunk,
    Document,
    Inventory,
    Table,
    by_element,
    scale_of,
    sentences,
    squeeze,
)
from aigent.retrieval.embed import Embedder
from aigent.retrieval.hits import Hit
from aigent.retrieval.rank import Indexes, Ranker, Reranked, Scope, build_ranker
from aigent.retrieval.rerank import LocalReranker, Reranker

Kind = Literal["chunk", "row", "table", "sentence"]

_DIGIT = re.compile(r"\d")
_LETTER = re.compile(r"[A-Za-z]")
_YEAR = re.compile(r"\b(?:19|20)\d\d(?:-\d\d)?\b|\bFY\s?\d\d(?:-\d\d)?\b")
_TABLE = "[Table"
_PAGE = re.compile(r"^\S+ p\.\d+$")  # "RELIANCE-FY25 p.100"


@dataclass(frozen=True)
class Keyed:
    """The keys as an inventory to index, the chunk id each stands for, and each key's kind."""

    keys: Inventory
    parent: Mapping[str, str]
    kind: Mapping[str, Kind]


def table_context(label: str) -> str:
    """A table label's section path and basis, without its report and page, scale or years."""
    parts = label.strip().removeprefix("[").removesuffix("]").split(" · ")
    return " · ".join(
        part
        for part in parts
        if part != "Table"
        and not _PAGE.match(part)
        and not part.startswith("columns:")
        and not scale_of(part)
    )


def column_headers(table: Table) -> list[str]:
    """Each column's heading: the distinct cells of its header rows, top to bottom."""
    width = max((len(row) for row in table.rows), default=0)
    headers: list[str] = []
    for column in range(width):
        cells: list[str] = []
        for row in table.rows[: table.header_rows]:
            cell = " ".join(row[column].split()) if column < len(row) else ""
            if cell and cell not in cells:
                cells.append(cell)
        headers.append(" ".join(cells))
    return headers


def years_in(headers: Sequence[str]) -> list[str]:
    """The years the column headings name, once each, in order: '2024-25', '2025', 'FY25'."""
    return list(dict.fromkeys(year for header in headers for year in _YEAR.findall(header)))


def row_label(row: Sequence[str]) -> str | None:
    """A row's first cell without a digit, when it has at least two letters; None for a row of
    figures or a placeholder such as '-' or '(e)'."""
    cells = [" ".join(cell.split()) for cell in row]
    label = next((cell for cell in cells if cell and not _DIGIT.search(cell)), None)
    return label if label is not None and len(_LETTER.findall(label)) >= 2 else None


def _body(chunk: Chunk) -> str:
    """A text chunk without the label line and the heading line that may open it."""
    body = chunk.text
    if body.startswith("[") and "]\n" in body:
        body = body.split("]\n", 1)[1]
    heading = (chunk.heading or "").split(" › ")[-1]
    if heading and body.startswith(f"{heading}\n"):
        body = body[len(heading) + 1 :]
    return body


def build_keys(documents: Sequence[Document], chunks: Inventory) -> Keyed:
    """Every chunk as a key of itself, then its row and table keys or, for text, its sentences.

    A row's key points at the piece of a split table that prints it, else the table's first piece.
    A text chunk of one sentence gets no sentence key: its own text already is one.
    """
    owned: dict[str, list[Chunk]] = {}
    for chunk in chunks.chunks:
        owned.setdefault(chunk.doc_id, []).append(chunk)
    keys: list[Chunk] = []
    parent: dict[str, str] = {}
    kind: dict[str, Kind] = {}
    count: dict[str, int] = {}

    def add(of: Chunk, text: str, key_kind: Kind) -> None:
        n = count.get(of.doc_id, 0)
        count[of.doc_id] = n + 1
        key = Chunk(
            id=f"{of.doc_id}#k{n:06d}", text=text, doc_id=of.doc_id, ordinal=n, start=of.start
        )
        keys.append(key)
        parent[key.id] = of.id
        kind[key.id] = key_kind

    for document in documents:
        own = owned.get(document.doc_id, [])
        for chunk in own:
            add(chunk, chunk.text, "chunk")
        for table in document.tables:
            pieces = [
                c for c in own if c.text.startswith(_TABLE) and table.start <= c.start < table.end
            ]
            if not pieces:
                continue
            context = table_context(pieces[0].text.split("\n", 1)[0])
            years = years_in(column_headers(table))
            when = f" · {', '.join(years)}" if years else ""
            printed = [squeeze(piece.text) for piece in pieces]
            labels: dict[str, list[str]] = {piece.id: [] for piece in pieces}
            for row in table.rows[table.header_rows :]:
                label = row_label(row)
                if label is None:
                    continue
                line = squeeze(" ".join(cell for cell in row if cell))
                home = next(
                    (p for p, text in zip(pieces, printed, strict=True) if line in text), pieces[0]
                )
                labels[home.id].append(label)
                add(home, f"{context} · {label}{when}", "row")
            for piece in pieces:
                if labels[piece.id]:
                    add(piece, f"{context}{when}\nrows: " + "; ".join(labels[piece.id]), "table")
        for chunk in own:
            if chunk.text.startswith(_TABLE):
                continue
            body = _body(chunk)
            spans = sentences(body)
            if len(spans) < 2:
                continue
            for start, end in spans:
                sentence = body[start:end].strip()
                add(
                    chunk,
                    f"{chunk.heading} · {sentence}" if chunk.heading else sentence,
                    "sentence",
                )

    inventory = Inventory(f"{chunks.name}-keys", tuple(keys), {key.id: key for key in keys})
    return Keyed(inventory, parent, kind)


@dataclass(frozen=True)
class ParentRanker:
    """Ranks the keys, then returns each chunk once, at its best key's place and score."""

    base: Ranker
    parent: Mapping[str, str]
    depth: int = KEY_DEPTH

    def rank(self, query: str, k: int = TOP_K, *, doc_id: Scope = None) -> list[Hit]:
        found: list[Hit] = []
        seen: set[str] = set()
        for hit in self.base.rank(query, max(self.depth, k), doc_id=doc_id):
            chunk_id = self.parent[hit.chunk_id]
            if chunk_id not in seen:
                seen.add(chunk_id)
                found.append(Hit(chunk_id, hit.score))
                if len(found) == k:
                    break
        return found


def keyed_search(
    name: str,
    documents: Sequence[Document],
    *,
    cache: bool,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    candidates: int = RERANK_CANDIDATES,
) -> tuple[Inventory, Ranker]:
    """`documents` chunked by element and indexed by key; the chunks come back by `SEARCH_METHOD`,
    the reranker, if any, reading each chunk whole. Loads the local models unless handed them."""
    chunks = Inventory.build(name, documents, by_element())
    keyed = build_keys(documents, chunks)
    indexes = Indexes.build(keyed.keys, embedder, cache=cache)
    base = build_ranker(SEARCH_METHOD.removesuffix("+rerank"), indexes, depth=KEY_DEPTH)
    ranker: Ranker = ParentRanker(base, keyed.parent)
    if SEARCH_METHOD.endswith("+rerank"):
        model = reranker if reranker is not None else LocalReranker()
        ranker = Reranked(ranker, model, chunks, candidates)
    return chunks, ranker
