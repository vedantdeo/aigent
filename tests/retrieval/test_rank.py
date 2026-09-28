"""Rankers: what `build_ranker` makes of each method, and what each ranker does with a scope.

The load-bearing test is the scoped one. Filtering the global top k by report returns nothing when
the other report ranks higher overall, which reads as "the report does not say" rather than as a
search that looked in the wrong place.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from aigent.config import FUSE_DEPTH, RERANK_CANDIDATES, SEARCH_METHOD
from aigent.retrieval.chunk import Document, Inventory, by_sentence
from aigent.retrieval.dense import DenseIndex
from aigent.retrieval.embed import Embedder
from aigent.retrieval.fuse import reciprocal_rank_fusion
from aigent.retrieval.hits import Hit, rank_ids
from aigent.retrieval.rank import (
    METHODS,
    HybridRanker,
    Indexes,
    Ranker,
    Reranked,
    build_ranker,
)
from aigent.retrieval.rerank import rerank

from ..conftest import BagOfWordsEmbedder, KeywordReranker

QUERY = "dividend policy"
DOCUMENTS = [
    Document(
        doc_id="ALPHA",
        text="The board recommended a final dividend of ten rupees per share. "
        "The company opened two new plants in Gujarat during the year. ",
    ),
    Document(
        doc_id="BETA",
        text="Dividend policy: the dividend payout target is forty per cent of profit. "
        "The dividend policy was reviewed and the dividend policy was kept. ",
    ),
]


def _indexes(embedder: BagOfWordsEmbedder) -> Indexes:
    return Indexes.build(Inventory.build("sentence", DOCUMENTS, by_sentence(80, 0)), embedder)


def _shape(ranker: Ranker) -> str:
    """A ranker written out as the composition it is."""
    if isinstance(ranker, Reranked):
        return f"Reranked({_shape(ranker.base)})"
    if isinstance(ranker, HybridRanker):
        return f"HybridRanker({_shape(ranker.dense)}, {_shape(ranker.sparse)})"
    return type(ranker).__name__


@pytest.mark.parametrize(
    ("method", "shape"),
    [
        ("dense", "DenseRanker"),
        ("dense+rerank", "Reranked(DenseRanker)"),
        ("bm25", "SparseRanker"),
        ("bm25+rerank", "Reranked(SparseRanker)"),
        ("hybrid", "HybridRanker(DenseRanker, SparseRanker)"),
        ("hybrid+rerank", "Reranked(HybridRanker(DenseRanker, SparseRanker))"),
    ],
)
def test_each_method_builds_the_composition_its_name_says(
    embedder: BagOfWordsEmbedder, method: str, shape: str
) -> None:
    """`+rerank` wraps whatever base it follows, and `hybrid` holds the other two bases: six
    methods out of three rankers and one wrapper."""
    assert _shape(build_ranker(method, _indexes(embedder), KeywordReranker())) == shape


@pytest.mark.parametrize("method", METHODS)
def test_a_scoped_ranking_ranks_within_its_report(
    embedder: BagOfWordsEmbedder, method: str
) -> None:
    indexes = _indexes(embedder)
    ranker = build_ranker(method, indexes, KeywordReranker())

    overall = ranker.rank(QUERY, k=1)
    scoped = ranker.rank(QUERY, k=2, doc_id="ALPHA")

    by_id = indexes.inventory.by_id
    assert by_id[overall[0].chunk_id].doc_id == "BETA", "the other report must rank higher overall"
    assert scoped and {by_id[hit.chunk_id].doc_id for hit in scoped} == {"ALPHA"}
    assert "dividend" in by_id[scoped[0].chunk_id].text


def _dense(indexes: Indexes, k: int) -> list[Hit]:
    return indexes.dense.search(indexes.embedder.embed_query(QUERY), k=k)


def _reranked(indexes: Indexes, candidates: list[Hit]) -> list[Hit]:
    texts = [indexes.inventory.by_id[hit.chunk_id].text for hit in candidates]
    return rerank(KeywordReranker(), QUERY, candidates, texts, k=3)


Expected = Callable[[Indexes], list[Hit]]


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        pytest.param("dense", lambda ix: _dense(ix, 3), id="dense is the dense index's own"),
        pytest.param(
            "bm25", lambda ix: ix.sparse.search(QUERY, k=3), id="bm25 is the sparse's own"
        ),
        pytest.param(
            "hybrid",
            lambda ix: reciprocal_rank_fusion(
                [rank_ids(_dense(ix, FUSE_DEPTH)), rank_ids(ix.sparse.search(QUERY, k=FUSE_DEPTH))],
                k=3,
            ),
            id="hybrid fuses dense then sparse, each read deep",
        ),
        pytest.param(
            "bm25+rerank",
            lambda ix: _reranked(ix, ix.sparse.search(QUERY, k=RERANK_CANDIDATES)),
            id="rerank reorders its base's candidates",
        ),
    ],
)
def test_an_unscoped_ranking_is_the_one_the_eval_measured(
    embedder: BagOfWordsEmbedder, method: str, expected: Expected
) -> None:
    """The ranking was restructured into classes; without a scope it must be exactly what it was,
    or every number measured through it describes a different retriever from the one in use."""
    indexes = _indexes(embedder)

    assert build_ranker(method, indexes, KeywordReranker()).rank(QUERY, 3) == expected(indexes)


def test_an_unknown_method_is_refused(embedder: BagOfWordsEmbedder) -> None:
    with pytest.raises(ValueError, match="unknown retrieval method"):
        build_ranker("cosine", _indexes(embedder))


def test_the_default_method_is_one_that_exists() -> None:
    assert SEARCH_METHOD in METHODS


@pytest.mark.parametrize(
    ("cache", "reused"),
    [
        pytest.param(False, False, id="not unless asked"),
        pytest.param(True, True, id="when asked"),
    ],
)
def test_indexes_reuse_stored_vectors_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, embedder: BagOfWordsEmbedder, cache: bool, reused: bool
) -> None:
    """Opt-in, as the corpus cache is: a stale hit is wrong quietly."""
    asked: list[str] = []

    def recorded(inventory: Inventory, embedder: Embedder) -> DenseIndex:
        asked.append(inventory.name)
        return DenseIndex.build(inventory, embedder)

    monkeypatch.setattr(DenseIndex, "build_cached", recorded)

    Indexes.build(Inventory.build("sentence", DOCUMENTS, by_sentence(80, 0)), embedder, cache=cache)

    assert bool(asked) is reused
