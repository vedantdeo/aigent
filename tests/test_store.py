"""The brute-force store: ranking, ties, and the two shapes it refuses to build from.

Runs against `BagOfWordsEmbedder`, so these test the store's arithmetic rather than any model's
quality.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from entropic.week02.chunk import Document, Inventory, by_sentence
from entropic.week02.embed import Vectors
from entropic.week02.store import VectorStore, rank_ids

from .conftest import FAKE_DIMENSIONS, BagOfWordsEmbedder

REVENUE = "Revenue for the year rose seven per cent on stronger retail volumes across the country. "
RISK = "Commodity price risk is hedged quarterly under a policy the board reviews every year now. "
DIVIDEND = "The board recommended a final dividend of ten rupees per equity share for the year. "


def _inventory(*paragraphs: str) -> Inventory:
    documents = [
        Document(doc_id=f"doc-{i}", text=text) for i, text in enumerate(paragraphs, start=1)
    ]
    return Inventory.build("sentence", documents, by_sentence(400, overlap_sentences=0))


def test_the_store_lines_its_ids_up_with_its_rows(embedder: BagOfWordsEmbedder) -> None:
    inventory = _inventory(REVENUE, RISK, DIVIDEND)

    store = VectorStore.build(inventory, embedder)

    assert store.ids == tuple(inventory.ids())
    assert store.matrix.shape == (len(inventory), FAKE_DIMENSIONS)
    assert store.embedder_name == "bag-of-words-fake"


def test_search_ranks_the_chunk_that_shares_the_query_words_first(
    embedder: BagOfWordsEmbedder,
) -> None:
    inventory = _inventory(REVENUE, RISK, DIVIDEND)
    store = VectorStore.build(inventory, embedder)

    hits = store.search(embedder.embed_query("dividend per equity share"), k=3)

    assert rank_ids(hits)[0] == "doc-3#0000", [(h.chunk_id, round(h.score, 3)) for h in hits]
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


@pytest.mark.parametrize(
    ("paragraphs", "k", "returned"),
    [
        pytest.param((REVENUE, RISK, DIVIDEND), 2, 2, id="k smaller than the inventory"),
        pytest.param((REVENUE, RISK), 5, 2, id="k larger than the inventory returns what exists"),
        pytest.param((), 5, 0, id="an empty store ranks nothing rather than raising"),
    ],
)
def test_search_returns_at_most_k_hits(
    embedder: BagOfWordsEmbedder, paragraphs: tuple[str, ...], k: int, returned: int
) -> None:
    store = VectorStore.build(_inventory(*paragraphs), embedder)

    assert len(store.search(embedder.embed_query("revenue"), k=k)) == returned


def test_ties_break_the_same_way_every_time_so_a_metric_can_be_compared(
    embedder: BagOfWordsEmbedder,
) -> None:
    """Two chunks with identical text score identically; the order between them must not wobble."""
    store = VectorStore.build(_inventory(REVENUE, REVENUE, REVENUE), embedder)
    query = embedder.embed_query("revenue for the year")

    orders = {tuple(rank_ids(store.search(query, k=3))) for _ in range(5)}

    assert orders == {("doc-1#0000", "doc-2#0000", "doc-3#0000")}


def test_the_store_refuses_vectors_it_cannot_treat_as_cosine() -> None:
    inventory = _inventory(REVENUE, RISK)

    with pytest.raises(ValueError, match="un-normalised"):
        VectorStore.build(inventory, BagOfWordsEmbedder(normalise=False))


def test_the_store_refuses_a_matrix_that_does_not_match_the_inventory(
    embedder: BagOfWordsEmbedder,
) -> None:
    """A silent misalignment returns real ids attached to another chunk's vector."""
    inventory = _inventory(REVENUE, RISK, DIVIDEND)

    class ShortEmbedder(BagOfWordsEmbedder):
        """One row short, the way a batching bug drops the last batch."""

        def embed_documents(self, texts: Sequence[str]) -> Vectors:
            return super().embed_documents(texts)[:-1]

    with pytest.raises(ValueError, match="2 vectors for 3 chunks"):
        VectorStore.build(inventory, ShortEmbedder())


def test_a_k_below_one_is_refused(embedder: BagOfWordsEmbedder) -> None:
    store = VectorStore.build(_inventory(REVENUE), embedder)

    with pytest.raises(ValueError, match="at least 1"):
        store.search(embedder.embed_query("revenue"), k=0)
