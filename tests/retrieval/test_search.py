"""The ranking `evaluate` scores and a caller searches: six methods, and a scope to one report.

The load-bearing test is the scoped one. Filtering the global top k by report returns nothing when
the other report ranks higher overall, which reads as "the report does not say" rather than as a
search that looked in the wrong place.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from entropic.config import FUSE_DEPTH, SEARCH_METHOD
from entropic.retrieval.chunk import Document, Inventory, by_sentence
from entropic.retrieval.embed import Vectors
from entropic.retrieval.fuse import reciprocal_rank_fusion
from entropic.retrieval.search import METHODS, Retrievers, Searcher
from entropic.retrieval.sparse import Bm25Index
from entropic.retrieval.store import Hit, VectorStore, rank_ids

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


def _retrievers(embedder: BagOfWordsEmbedder, *, reranker: bool = True) -> Retrievers:
    inventory = Inventory.build("sentence", DOCUMENTS, by_sentence(80, 0))
    return Retrievers(
        inventory=inventory,
        store=VectorStore.build(inventory, embedder),
        index=Bm25Index.build(inventory),
        reranker=KeywordReranker() if reranker else None,
    )


@pytest.mark.parametrize("method", METHODS)
def test_a_scoped_search_ranks_within_its_report(embedder: BagOfWordsEmbedder, method: str) -> None:
    retrievers = _retrievers(embedder)
    vector = embedder.embed_query(QUERY)

    overall = retrievers.rank(method, QUERY, vector, k=1)
    scoped = retrievers.rank(method, QUERY, vector, k=2, doc_id="ALPHA")

    by_id = retrievers.inventory.by_id
    assert by_id[overall[0].chunk_id].doc_id == "BETA", "the other report must rank higher overall"
    assert scoped and {by_id[hit.chunk_id].doc_id for hit in scoped} == {"ALPHA"}
    assert "dividend" in by_id[scoped[0].chunk_id].text


Unscoped = Callable[[Retrievers, Vectors], list[Hit]]


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        pytest.param("dense", lambda r, v: r.store.search(v, k=3), id="dense is the store's own"),
        pytest.param("bm25", lambda r, v: r.index.search(QUERY, k=3), id="bm25 is the index's own"),
        pytest.param(
            "hybrid",
            lambda r, v: reciprocal_rank_fusion(
                [
                    rank_ids(r.store.search(v, k=FUSE_DEPTH)),
                    rank_ids(r.index.search(QUERY, k=FUSE_DEPTH)),
                ],
                k=3,
            ),
            id="hybrid fuses dense then lexical",
        ),
    ],
)
def test_an_unscoped_ranking_is_the_one_the_eval_measured(
    embedder: BagOfWordsEmbedder, method: str, expected: Unscoped
) -> None:
    """The ranking moved here from `evaluate`; without a scope it must be exactly what it was, or
    every number measured through it describes a different retriever from the one in use."""
    retrievers = _retrievers(embedder)
    vector = embedder.embed_query(QUERY)

    assert retrievers.rank(method, QUERY, vector, k=3) == expected(retrievers, vector)


@pytest.mark.parametrize(
    ("method", "says"),
    [
        pytest.param("cosine", "unknown retrieval method", id="a method that does not exist"),
        pytest.param("dense+rerank", "needs a reranker", id="a rerank with no reranker"),
    ],
)
def test_a_method_that_cannot_run_is_refused(
    embedder: BagOfWordsEmbedder, method: str, says: str
) -> None:
    retrievers = _retrievers(embedder, reranker=False)

    with pytest.raises(ValueError, match=says):
        retrievers.rank(method, QUERY, embedder.embed_query(QUERY))


def test_the_searcher_returns_passages_and_honours_the_scope(embedder: BagOfWordsEmbedder) -> None:
    searcher = Searcher.build(DOCUMENTS, embedder=embedder, reranker=KeywordReranker())

    assert searcher.method == SEARCH_METHOD
    assert [chunk.doc_id for chunk in searcher.search(QUERY, k=1)] == ["BETA"]
    assert [chunk.doc_id for chunk in searcher.search(QUERY, k=1, doc_id="ALPHA")] == ["ALPHA"]
