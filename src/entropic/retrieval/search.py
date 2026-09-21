"""One inventory, ranked by any of six methods: what `evaluate` scores, and what a caller searches.

The ranking lives here rather than in `evaluate`, so the code that is measured and the code that
is used are the same code. `doc_id` scopes a search to one report without a second index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from entropic.config import FUSE_DEPTH, RERANK_CANDIDATES, SEARCH_METHOD, TOP_K
from entropic.retrieval.chunk import Chunk, Document, Inventory, by_sentence
from entropic.retrieval.embed import Embedder, LocalEmbedder, Vectors
from entropic.retrieval.fuse import reciprocal_rank_fusion
from entropic.retrieval.rerank import LocalReranker, Reranker, rerank
from entropic.retrieval.sparse import Bm25Index
from entropic.retrieval.store import Hit, VectorStore, rank_ids

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
METHOD_STRATEGY = "by_sentence"


@dataclass(frozen=True)
class Retrievers:
    """Everything the six methods rank with, built once over one inventory.

    `reranker` is None until a method needs one: it is a second local model to load, and the
    cheap methods must not pay for it.
    """

    inventory: Inventory
    store: VectorStore
    index: Bm25Index
    reranker: Reranker | None = None

    def rank(
        self,
        method: str,
        question: str,
        vector: Vectors,
        k: int = TOP_K,
        *,
        depth: int = FUSE_DEPTH,
        candidates: int = RERANK_CANDIDATES,
        doc_id: str | None = None,
    ) -> list[Hit]:
        """The best `k` chunks by one method. A `+rerank` method rescores `candidates` from its
        base and keeps `k`, so the base decides reach and the reranker only decides order."""
        if method not in METHODS:
            raise ValueError(f"unknown retrieval method {method!r}; expected one of {METHODS}")
        base, reranked = method.removesuffix("+rerank"), method.endswith("+rerank")
        n = candidates if reranked else k
        hits = self._first_stage(base, question, vector, n, depth, doc_id)
        if not reranked:
            return hits
        if self.reranker is None:
            raise ValueError(f"{method} needs a reranker and was given none")
        texts = [self.inventory.by_id[hit.chunk_id].text for hit in hits]
        return rerank(self.reranker, question, hits, texts, k=k)

    def _first_stage(
        self, base: str, question: str, vector: Vectors, n: int, depth: int, doc_id: str | None
    ) -> list[Hit]:
        if base != "hybrid":
            return self._ranked(base, question, vector, n, doc_id)
        # Fuse the top `depth` of each ranking, not its head: rank 20 lexically and rank 4 densely
        # is the chunk fusion exists to promote.
        return reciprocal_rank_fusion(
            [
                rank_ids(self._ranked("dense", question, vector, depth, doc_id)),
                rank_ids(self._ranked("bm25", question, vector, depth, doc_id)),
            ],
            k=n,
        )

    def _ranked(
        self, ranker: str, question: str, vector: Vectors, n: int, doc_id: str | None
    ) -> list[Hit]:
        # Scoped, rank everything and drop the other reports, so a report that ranks low overall
        # still returns its own best n.
        reach = n if doc_id is None else max(len(self.inventory), 1)
        if ranker == "bm25":
            hits = self.index.search(question, k=reach)
        else:
            hits = self.store.search(vector, k=reach)
        if doc_id is None:
            return hits
        return [hit for hit in hits if self.inventory.by_id[hit.chunk_id].doc_id == doc_id][:n]


@dataclass(frozen=True)
class Searcher:
    """A query in, passages out: one method, fixed, over one inventory."""

    retrievers: Retrievers
    embedder: Embedder
    method: str = SEARCH_METHOD

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        method: str = SEARCH_METHOD,
    ) -> Searcher:
        """Chunk by sentence, embed and index. Loads the local models unless handed stand-ins."""
        if embedder is None:
            embedder = LocalEmbedder()
        if reranker is None and method.endswith("+rerank"):
            reranker = LocalReranker()
        inventory = Inventory.build(METHOD_STRATEGY, documents, by_sentence())
        retrievers = Retrievers(
            inventory=inventory,
            store=VectorStore.build(inventory, embedder),
            index=Bm25Index.build(inventory),
            reranker=reranker,
        )
        return cls(retrievers=retrievers, embedder=embedder, method=method)

    def search(self, query: str, k: int = TOP_K, *, doc_id: str | None = None) -> list[Chunk]:
        """The `k` passages that best answer `query`, from one report when `doc_id` is given."""
        vector = self.embedder.embed_query(query)
        hits = self.retrievers.rank(self.method, query, vector, k, doc_id=doc_id)
        return [self.retrievers.inventory.by_id[hit.chunk_id] for hit in hits]


def context_block(chunks: Sequence[Chunk]) -> str:
    """The retrieved passages, each tagged with the id the model must cite it by.

    XML-delimited and one passage per element, so the model can point at one of them rather than
    at the blob — a citation that names no passage cannot be checked against the label.
    """
    return "\n\n".join(
        f'<passage id="{chunk.id}" source="{chunk.citation}">\n{chunk.text}\n</passage>'
        for chunk in chunks
    )
