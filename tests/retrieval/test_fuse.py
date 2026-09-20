"""Reciprocal rank fusion, which is arithmetic over positions and nothing else.

The load-bearing test is the damping one: it is the only thing standing between "hybrid retrieval"
and "whichever ranker shouted first wins", and it is a property of the constant rather than of any
model, so it can be asserted exactly.
"""

from __future__ import annotations

import pytest

from entropic.retrieval.fuse import reciprocal_rank_fusion
from entropic.retrieval.store import rank_ids


def test_a_chunk_both_rankers_found_beats_one_only_the_first_put_top() -> None:
    """The whole point of fusing, and the realistic shape: two rankers returning mostly different
    ids. `b` is the only chunk in both lists, at rank 2 and rank 3, and it beats `a` and `x` which
    are each first in one list and absent from the other.

    Note what this does *not* claim. Given equal rank sums RRF prefers the more lopsided pair —
    `1/(k+r)` is convex, so (1,3) scores above (2,2) — which is why the case worth testing is
    presence in both rankings rather than a tidy symmetric example."""
    dense = ["a", "b", "c"]
    lexical = ["x", "y", "b"]

    fused = rank_ids(reciprocal_rank_fusion([dense, lexical], k=5))

    assert fused[0] == "b", fused


def test_damping_is_what_stops_one_ranker_dominating() -> None:
    """At damping 0 a first place scores 1.0 against a second place's 0.5, so being top of one
    list beats being near the top of both. At 60 the two are 1/61 and 1/62 and agreement wins.
    Same inputs, opposite answers — this is the constant doing the work, not the model."""
    dense = ["a", "b"]
    lexical = ["c", "d", "b"]

    loud = rank_ids(reciprocal_rank_fusion([dense, lexical], k=4, damping=0))
    damped = rank_ids(reciprocal_rank_fusion([dense, lexical], k=4, damping=60))

    assert loud[0] != "b", f"at damping 0 a first place wins outright: {loud}"
    assert damped[0] == "b", f"at damping 60 being in both lists wins: {damped}"


def test_a_chunk_only_one_ranker_found_still_appears() -> None:
    """Fusion must not be an intersection: the lexical ranker finding a rare token the dense one
    missed entirely is the case hybrid retrieval is bought for."""
    fused = rank_ids(reciprocal_rank_fusion([["a", "b"], ["z"]], k=5))

    assert set(fused) == {"a", "b", "z"}


def test_a_repeat_inside_one_ranking_counts_once_at_its_best_rank() -> None:
    """A ranker returning the same id twice has found one thing; counting it twice would let a
    ranker vote for a chunk as often as it liked."""
    once = reciprocal_rank_fusion([["a", "b"]], k=5)
    twice = reciprocal_rank_fusion([["a", "b", "a"]], k=5)

    assert [(h.chunk_id, h.score) for h in once] == [(h.chunk_id, h.score) for h in twice]


def test_ties_break_by_first_appearance_so_two_identical_runs_agree() -> None:
    """`a` and `b` are symmetric here, so only a stable rule decides. Without one the order comes
    from whatever the dict happened to do, which is the same wobble `VectorStore.search` refuses."""
    for _ in range(5):
        assert rank_ids(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=2)) == ["a", "b"]


def test_fusing_one_ranking_preserves_its_order() -> None:
    """The degenerate case has to be the identity, or `hybrid` with one ranker disabled would not
    reduce to that ranker and the comparison between arms would mean nothing."""
    assert rank_ids(reciprocal_rank_fusion([["a", "b", "c"]], k=3)) == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("rankings", "expected"),
    [
        pytest.param([], [], id="no rankers at all"),
        pytest.param([[], []], [], id="two rankers that found nothing"),
        pytest.param([[], ["a"]], ["a"], id="one empty, one not"),
    ],
)
def test_fusion_of_nothing_is_nothing(rankings: list[list[str]], expected: list[str]) -> None:
    assert rank_ids(reciprocal_rank_fusion(rankings, k=5)) == expected


@pytest.mark.parametrize(
    ("kwargs", "says"),
    [
        pytest.param({"k": 0}, "at least 1", id="k below one"),
        pytest.param({"damping": -1}, "not be negative", id="negative damping"),
    ],
)
def test_bad_parameters_are_refused(kwargs: dict[str, int], says: str) -> None:
    with pytest.raises(ValueError, match=says):
        reciprocal_rank_fusion([["a"]], **kwargs)
