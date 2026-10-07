"""What every pattern works over: the three annual reports, and the search that reads them."""

from __future__ import annotations

import json
from typing import Literal, Protocol, cast

from aigent.config import SEARCH_METHOD
from aigent.retrieval.chunk import Chunk, Inventory, by_sentence
from aigent.retrieval.corpus import MANIFEST, load_corpus
from aigent.retrieval.rank import Indexes, build_ranker

# As a schema field, this holds the model to a real report. A test holds it to the manifest.
DocId = Literal["ITC-FY25", "RELIANCE-FY25", "TATAMOTORS-FY25"]

_DOCUMENTS = cast(
    dict[str, dict[str, object]], json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
)
REPORTS: dict[str, str] = {doc_id: str(entry["company"]) for doc_id, entry in _DOCUMENTS.items()}
CATALOGUE = "\n".join(f"- {doc_id}: {company}" for doc_id, company in REPORTS.items())


class Search(Protocol):
    """A query in, passages out, from one report when `doc_id` is given."""

    def __call__(self, query: str, /, *, doc_id: str | None = None) -> list[Chunk]: ...


def build_search(cache: bool) -> Search:
    """Chunk, embed and index the corpus once; passages come back by `SEARCH_METHOD`."""
    inventory = Inventory.build("by_sentence", load_corpus(cache=cache), by_sentence())
    ranker = build_ranker(SEARCH_METHOD, Indexes.build(inventory, cache=cache))

    def search(query: str, /, *, doc_id: str | None = None) -> list[Chunk]:
        return [inventory.by_id[hit.chunk_id] for hit in ranker.rank(query, doc_id=doc_id)]

    return search
