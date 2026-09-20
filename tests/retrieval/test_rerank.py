"""Cross-encoder reranking, against a fake that deliberately disagrees with the fake retriever.

No torch here: `rerank` takes a `Reranker` protocol and the candidate texts, so everything except
loading the model is testable against plain strings. The load-bearing test is the last one — a
reranker cannot retrieve, and a suite that never said so would let someone read a rerank column as
if it fixed misses.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from entropic.retrieval.rerank import rerank
from entropic.retrieval.store import Hit, rank_ids

from ..conftest import KeywordReranker

CANDIDATES = [Hit("c1", 0.9), Hit("c2", 0.8), Hit("c3", 0.7)]
TEXTS = [
    "the board reviewed its financing arrangements",
    "a final dividend of ten rupees per equity share",
    "the company continued its normal course of business",
]


def test_reranking_reorders_what_the_first_stage_ranked_badly() -> None:
    """`c2` arrived last on cosine and holds the answer. Reordering it to the front is the entire
    job: the first stage decides what is *reachable*, the reranker decides what is *first*."""
    reranked = rerank(KeywordReranker(), "final dividend equity share", CANDIDATES, TEXTS, k=3)

    assert rank_ids(reranked)[0] == "c2", rank_ids(reranked)


def test_the_reranker_sees_the_query_with_every_candidate_in_one_call() -> None:
    """A cross-encoder is one forward pass per pair, which is why it runs over tens of candidates
    and never over the corpus. Batching them into one call is what keeps that affordable."""
    fake = KeywordReranker()

    rerank(fake, "dividend", CANDIDATES, TEXTS, k=2)

    assert fake.seen == [("dividend", 3)], "one call, every candidate, not one call each"


def test_only_k_survive_but_the_cut_is_made_after_rescoring() -> None:
    """Cutting before the rerank would make the reranker's opinion of the tail unreachable, which
    is the bug that makes a rerank arm look like it did nothing."""
    reranked = rerank(KeywordReranker(), "final dividend equity share", CANDIDATES, TEXTS, k=1)

    assert rank_ids(reranked) == ["c2"]


def test_ties_break_by_the_order_the_first_stage_produced() -> None:
    """The cheap ranker settles what the expensive one is indifferent about. Nothing in these
    passages matches, so every score is zero and only the incoming order can decide."""
    reranked = rerank(KeywordReranker(), "cryptocurrency", CANDIDATES, TEXTS, k=3)

    assert rank_ids(reranked) == ["c1", "c2", "c3"]


def test_no_candidates_is_no_call_and_no_result() -> None:
    fake = KeywordReranker()

    assert rerank(fake, "dividend", [], [], k=5) == []
    assert fake.seen == [], "an empty candidate list must not cost a forward pass"


class _Miscounting:
    """A reranker that returns the wrong number of scores, as a batching bug would."""

    name = "miscounting-fake"

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        del query
        return [1.0] * (len(passages) - 1)


@pytest.mark.parametrize(
    ("texts", "k", "says"),
    [
        pytest.param(TEXTS[:2], 3, "2 texts for 3 candidates", id="texts misaligned"),
        pytest.param(TEXTS, 0, "at least 1", id="k below one"),
    ],
)
def test_a_misaligned_call_is_refused_rather_than_scored(
    texts: list[str], k: int, says: str
) -> None:
    """Ids and texts aligned by position is the contract; one short and every citation names the
    wrong passage while every score still looks plausible. Same shape as the store's row check."""
    with pytest.raises(ValueError, match=says):
        rerank(KeywordReranker(), "dividend", CANDIDATES, texts, k=k)


def test_a_reranker_returning_too_few_scores_is_caught() -> None:
    with pytest.raises(ValueError, match="2 scores for 3 candidates"):
        rerank(_Miscounting(), "dividend", CANDIDATES, TEXTS, k=3)


def test_reranking_cannot_retrieve_what_the_first_stage_missed() -> None:
    """The limitation worth a test of its own. A rerank column moves MRR and precision; it moves
    recall only as far as a wider candidate pool does, and reading it otherwise would credit the
    reranker for retrieval's work."""
    without = rerank(KeywordReranker(), "final dividend", CANDIDATES[:1], TEXTS[:1], k=3)

    assert rank_ids(without) == ["c1"], "the answer chunk was never a candidate, so it cannot win"
