"""Rankers: a query in, the best chunks of one inventory out, by any of six methods.

`build_ranker` turns a method name into a ranker, and every ranker answers `rank`. `hybrid` holds a
dense and a sparse ranker and fuses them; a `+rerank` method is its base wrapped in `Reranked`. The
indexes are built once, in `Indexes`, and shared: embedding the corpus is the slow part.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from entropic.config import FUSE_DEPTH, RERANK_CANDIDATES, TOP_K
from entropic.retrieval.chunk import Inventory
from entropic.retrieval.dense import DenseIndex
from entropic.retrieval.embed import Embedder, LocalEmbedder
from entropic.retrieval.fuse import reciprocal_rank_fusion
from entropic.retrieval.hits import Hit, rank_ids
from entropic.retrieval.rerank import LocalReranker, Reranker, rerank
from entropic.retrieval.sparse import SparseIndex

# Three bases, each with and without a cross-encoder over its own candidates, so a table of them
# separates what reach buys from what ordering buys.
METHODS = (
    "dense",
    "dense+rerank",
    "bm25",
    "bm25+rerank",
    "hybrid",
    "hybrid+rerank",
)


class Ranker(Protocol):
    """Anything that ranks an inventory's chunks for a query, best first."""

    def rank(self, query: str, k: int = TOP_K, *, doc_id: str | None = None) -> list[Hit]: ...


@dataclass(frozen=True)
class Indexes:
    """One inventory, embedded and indexed once, shared by every ranker built over it."""

    inventory: Inventory
    embedder: Embedder
    dense: DenseIndex
    sparse: SparseIndex

    @classmethod
    def build(cls, inventory: Inventory, embedder: Embedder | None = None) -> Indexes:
        """Embed and index `inventory`. Loads the local embedder unless handed one."""
        if embedder is None:
            embedder = LocalEmbedder()
        dense = DenseIndex.build(inventory, embedder)
        return cls(inventory, embedder, dense, SparseIndex.build(inventory))


def _scoped(
    search: Callable[[int], list[Hit]], k: int, doc_id: str | None, inventory: Inventory
) -> list[Hit]:
    """`search`'s best `k`, or with `doc_id` the best `k` from that report alone."""
    if doc_id is None:
        return search(k)
    # Rank everything and drop the other reports: the global top k filtered by report comes back
    # empty for a report that ranks low overall.
    everything = search(max(len(inventory), 1))
    return [hit for hit in everything if inventory.by_id[hit.chunk_id].doc_id == doc_id][:k]


@dataclass(frozen=True)
class DenseRanker:
    """The query's embedding against every chunk's, by cosine."""

    indexes: Indexes

    def rank(self, query: str, k: int = TOP_K, *, doc_id: str | None = None) -> list[Hit]:
        vector = self.indexes.embedder.embed_query(query)
        dense = self.indexes.dense
        return _scoped(lambda n: dense.search(vector, k=n), k, doc_id, self.indexes.inventory)


@dataclass(frozen=True)
class SparseRanker:
    """BM25 over the words the query shares with each chunk."""

    indexes: Indexes

    def rank(self, query: str, k: int = TOP_K, *, doc_id: str | None = None) -> list[Hit]:
        sparse = self.indexes.sparse
        return _scoped(lambda n: sparse.search(query, k=n), k, doc_id, self.indexes.inventory)


@dataclass(frozen=True)
class HybridRanker:
    """Two rankers fused by reciprocal rank, each read to `depth` rather than to `k`.

    Rank 20 lexically and rank 4 densely is the chunk fusion exists to promote, so it reads deep.
    """

    dense: Ranker
    sparse: Ranker
    depth: int = FUSE_DEPTH

    def rank(self, query: str, k: int = TOP_K, *, doc_id: str | None = None) -> list[Hit]:
        rankings = [
            rank_ids(self.dense.rank(query, self.depth, doc_id=doc_id)),
            rank_ids(self.sparse.rank(query, self.depth, doc_id=doc_id)),
        ]
        return reciprocal_rank_fusion(rankings, k=k)


@dataclass(frozen=True)
class Reranked:
    """Any ranker, with a cross-encoder reordering its best `candidates` and keeping `k`.

    The base decides what is reachable; the reranker only decides the order within it.
    """

    base: Ranker
    reranker: Reranker
    inventory: Inventory
    candidates: int = RERANK_CANDIDATES

    def rank(self, query: str, k: int = TOP_K, *, doc_id: str | None = None) -> list[Hit]:
        hits = self.base.rank(query, self.candidates, doc_id=doc_id)
        texts = [self.inventory.by_id[hit.chunk_id].text for hit in hits]
        return rerank(self.reranker, query, hits, texts, k=k)


def build_ranker(
    method: str,
    indexes: Indexes,
    reranker: Reranker | None = None,
    *,
    depth: int = FUSE_DEPTH,
    candidates: int = RERANK_CANDIDATES,
) -> Ranker:
    """The ranker `method` names, over `indexes`. A `+rerank` method loads the local cross-encoder
    unless handed a reranker."""
    if method not in METHODS:
        raise ValueError(f"unknown retrieval method {method!r}; expected one of {METHODS}")
    base = method.removesuffix("+rerank")
    ranker: Ranker
    if base == "dense":
        ranker = DenseRanker(indexes)
    elif base == "bm25":
        ranker = SparseRanker(indexes)
    else:
        ranker = HybridRanker(DenseRanker(indexes), SparseRanker(indexes), depth)
    if not method.endswith("+rerank"):
        return ranker
    reranker = reranker if reranker is not None else LocalReranker()
    return Reranked(ranker, reranker, indexes.inventory, candidates)
