"""Cross-encoder reranking: read the query and the passage together, then reorder.

`LocalEmbedder` is a bi-encoder — it encodes the query and the passage separately and never sees
them side by side. That is what lets every chunk be embedded once, ahead of time, and it is also
what makes it blunt: nothing in a passage's vector can depend on the question being asked.

A cross-encoder takes `(query, passage)` as one input and runs attention across both, so it can
notice that the question asks about FY25 and the passage is about FY24. It cannot be precomputed —
it is one forward pass per candidate, at query time — so it runs over the tens of chunks a cheap
retriever surfaced, never over the corpus.

**Reranking cannot fix a miss.** A chunk outside the candidate list cannot be reordered into it,
so this moves MRR and precision, and moves recall only as far as widening `RERANK_CANDIDATES` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from entropic.config import RERANK_BATCH, RERANK_MODEL, TOP_K
from entropic.retrieval.hits import Hit


class Reranker(Protocol):
    """What reranking needs from a model, and nothing more.

    A protocol for the same reason `Embedder` is one: a test can substitute a fake ordering
    without importing torch, and a hosted reranker drops in later as another column.
    """

    name: str

    def scores(self, query: str, passages: Sequence[str]) -> list[float]: ...


class LocalReranker:
    """A `sentence-transformers` CrossEncoder on whatever device this machine has.

    Heavy imports are deferred into `__init__`, as in `embed`, so importing this module costs
    nothing to anything that does not build one.
    """

    def __init__(self, model_name: str = RERANK_MODEL, *, device: str | None = None) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.name = model_name
        self.device = device
        self.batch_size = RERANK_BATCH
        self._model = CrossEncoder(model_name, device=device)

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage, in the order given.

        Unbounded logits, not probabilities and not comparable across models — only their order
        within one call means anything, which is all reranking uses.
        """
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        return [
            float(score)
            for score in cast(
                Sequence[float], self._model.predict(pairs, batch_size=self.batch_size)
            )
        ]


def rerank(
    reranker: Reranker,
    query: str,
    candidates: Sequence[Hit],
    texts: Sequence[str],
    k: int = TOP_K,
) -> list[Hit]:
    """Reorder candidates by the cross-encoder, best first, and keep the top k.

    `texts` is the passage text for each candidate, in the same order — the caller looks them up,
    so this module never needs an `Inventory` and stays testable against plain strings.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if len(texts) != len(candidates):
        raise ValueError(f"{len(texts)} texts for {len(candidates)} candidates")
    if not candidates:
        return []
    scored = reranker.scores(query, texts)
    if len(scored) != len(candidates):
        raise ValueError(f"reranker returned {len(scored)} scores for {len(candidates)} candidates")
    # Ties break by the order the first stage produced, so the cheap ranker settles what the
    # expensive one is indifferent about rather than letting the sort decide.
    order = sorted(range(len(candidates)), key=lambda row: (-scored[row], row))
    return [Hit(candidates[row].chunk_id, scored[row]) for row in order[:k]]
