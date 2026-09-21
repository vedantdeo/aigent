"""What every ranker returns: chunk ids, best first, each with the score that placed it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """One ranked chunk id and its score. Scores mean different things per ranker — cosine, BM25,
    fused rank, cross-encoder logit — so compare them only within one ranking."""

    chunk_id: str
    score: float


def rank_ids(hits: Sequence[Hit]) -> list[str]:
    """Just the ids, in rank order — the shape the retrieval graders read."""
    return [hit.chunk_id for hit in hits]
