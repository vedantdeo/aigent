"""Reciprocal rank fusion: several rankings of the same ids into one.

The scores coming out of the dense index and out of BM25 are not comparable. Cosine lives in
[-1, 1]; a BM25 score is unbounded and depends on the corpus it was computed over. Averaging them
is meaningless and normalising them is a tuning problem that has to be redone per corpus.

Rank is the one thing both rankers mean the same way, so fusion uses only that.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from aigent.config import RRF_K, TOP_K
from aigent.retrieval.hits import Hit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = TOP_K, *, damping: int = RRF_K
) -> list[Hit]:
    """Fuse ranked id lists into one, best first.

    Each ranking contributes `1 / (damping + rank)` to every id it lists, rank counting from 1.
    `damping` is doing real work at its default of 60: without it rank 1 scores 1.0 against rank
    2's 0.5, so whichever ranker happened to put something first would dominate the result. At 60
    the two are 1/61 and 1/62, so **agreement between rankers outweighs being first in one of
    them** — which is the whole behaviour hybrid retrieval is bought for.

    The fused score is not a similarity and must not be read as one. It says where the rankers put
    a chunk, not how much they liked it.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if damping < 0:
        raise ValueError(f"damping must not be negative, got {damping}")

    fused: dict[str, float] = defaultdict(float)
    # Insertion order of first appearance, so ties break the same way on two identical runs
    # rather than however the dict happened to be built.
    seen: list[str] = []
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            if chunk_id not in fused:
                seen.append(chunk_id)
            fused[chunk_id] += 1 / (damping + rank)

    order = {chunk_id: position for position, chunk_id in enumerate(seen)}
    ranked = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], order[chunk_id]))
    return [Hit(chunk_id, fused[chunk_id]) for chunk_id in ranked[:k]]
