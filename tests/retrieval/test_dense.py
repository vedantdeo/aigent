"""The dense index, brute-force cosine: ranking, ties, the two shapes it refuses to build from,
and the copy of it kept on disk.

Runs against `BagOfWordsEmbedder`, so these test the index's arithmetic rather than any model's
quality.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from aigent.retrieval.chunk import Document, Inventory, by_sentence
from aigent.retrieval.dense import DenseIndex, cache_file
from aigent.retrieval.embed import Vectors
from aigent.retrieval.hits import rank_ids

from ..conftest import FAKE_DIMENSIONS, BagOfWordsEmbedder

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

    index = DenseIndex.build(inventory, embedder)

    assert index.ids == tuple(inventory.ids())
    assert index.matrix.shape == (len(inventory), FAKE_DIMENSIONS)
    assert index.embedder_name == "bag-of-words-fake"


def test_search_ranks_the_chunk_that_shares_the_query_words_first(
    embedder: BagOfWordsEmbedder,
) -> None:
    inventory = _inventory(REVENUE, RISK, DIVIDEND)
    index = DenseIndex.build(inventory, embedder)

    hits = index.search(embedder.embed_query("dividend per equity share"), k=3)

    assert rank_ids(hits)[0] == "doc-3#0000", [(h.chunk_id, round(h.score, 3)) for h in hits]
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


@pytest.mark.parametrize(
    ("paragraphs", "k", "returned"),
    [
        pytest.param((REVENUE, RISK, DIVIDEND), 2, 2, id="k smaller than the inventory"),
        pytest.param((REVENUE, RISK), 5, 2, id="k larger than the inventory returns what exists"),
        pytest.param((), 5, 0, id="an empty index ranks nothing rather than raising"),
    ],
)
def test_search_returns_at_most_k_hits(
    embedder: BagOfWordsEmbedder, paragraphs: tuple[str, ...], k: int, returned: int
) -> None:
    index = DenseIndex.build(_inventory(*paragraphs), embedder)

    assert len(index.search(embedder.embed_query("revenue"), k=k)) == returned


def test_ties_break_the_same_way_every_time_so_a_metric_can_be_compared(
    embedder: BagOfWordsEmbedder,
) -> None:
    """Two chunks with identical text score identically; the order between them must not wobble."""
    index = DenseIndex.build(_inventory(REVENUE, REVENUE, REVENUE), embedder)
    query = embedder.embed_query("revenue for the year")

    orders = {tuple(rank_ids(index.search(query, k=3))) for _ in range(5)}

    assert orders == {("doc-1#0000", "doc-2#0000", "doc-3#0000")}


def test_the_store_refuses_vectors_it_cannot_treat_as_cosine() -> None:
    inventory = _inventory(REVENUE, RISK)

    with pytest.raises(ValueError, match="un-normalised"):
        DenseIndex.build(inventory, BagOfWordsEmbedder(normalise=False))


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
        DenseIndex.build(inventory, ShortEmbedder())


def test_a_k_below_one_is_refused(embedder: BagOfWordsEmbedder) -> None:
    index = DenseIndex.build(_inventory(REVENUE), embedder)

    with pytest.raises(ValueError, match="at least 1"):
        index.search(embedder.embed_query("revenue"), k=0)


# --- On disk ------------------------------------------------------------------------------------


class CountingEmbedder(BagOfWordsEmbedder):
    """Counts the passages it embeds, so a test can tell a cache hit from a rebuild."""

    def __init__(self, name: str = "bag-of-words-fake") -> None:
        super().__init__()
        self.name = name
        self.embedded = 0

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.embedded += len(texts)
        return super().embed_documents(texts)


class PrefixingEmbedder(CountingEmbedder):
    """The same model, embedding passages differently: what an edit to `embed_documents` does."""

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return super().embed_documents([f"passage: {text}" for text in texts])


def test_an_index_comes_back_from_disk_as_it_went_in(
    tmp_path: Path, embedder: BagOfWordsEmbedder
) -> None:
    index = DenseIndex.build(_inventory(REVENUE, RISK, DIVIDEND), embedder)
    path = tmp_path / "index.npz"

    index.save(path)
    loaded = DenseIndex.load(path)

    assert loaded.ids == index.ids
    assert np.array_equal(loaded.matrix, index.matrix)
    assert loaded.embedder_name == index.embedder_name
    query = embedder.embed_query("dividend per equity share")
    assert loaded.search(query) == index.search(query)
    assert [path.name for path in tmp_path.iterdir()] == ["index.npz"], "a partial file was left"


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        pytest.param(np.ones((2, 4), dtype=np.float32), "un-normalised", id="un-normalised rows"),
        pytest.param(np.eye(4, dtype=np.float32)[:1], "1 vectors for 2 chunks", id="a row short"),
    ],
)
def test_a_loaded_index_is_held_to_the_checks_a_built_one_is(
    tmp_path: Path, matrix: Vectors, match: str
) -> None:
    """The constructor checks nothing, so it can write the file a bug would."""
    path = tmp_path / "index.npz"
    DenseIndex(ids=("doc-1#0000", "doc-2#0000"), matrix=matrix).save(path)

    with pytest.raises(ValueError, match=match):
        DenseIndex.load(path)


def test_a_second_build_is_served_from_the_cache_without_embedding_again(tmp_path: Path) -> None:
    inventory = _inventory(REVENUE, RISK, DIVIDEND)
    embedder = CountingEmbedder()

    first = DenseIndex.build_cached(inventory, embedder, tmp_path)
    second = DenseIndex.build_cached(inventory, embedder, tmp_path)

    assert embedder.embedded == len(inventory), "the second build embedded the corpus again"
    assert second.ids == first.ids
    assert np.array_equal(second.matrix, first.matrix)


@pytest.mark.parametrize(
    ("paragraphs", "embedder", "embeds_again"),
    [
        pytest.param((REVENUE, RISK), CountingEmbedder(), False, id="nothing changed"),
        # Same ids, different text: the case a cache keyed on ids alone serves stale.
        pytest.param((REVENUE, DIVIDEND), CountingEmbedder(), True, id="a chunk's text changed"),
        pytest.param((REVENUE, RISK, DIVIDEND), CountingEmbedder(), True, id="a chunk was added"),
        pytest.param(
            (REVENUE, RISK), CountingEmbedder("another-model"), True, id="another embedding model"
        ),
        pytest.param(
            (REVENUE, RISK), PrefixingEmbedder(), True, id="the code that embeds a passage changed"
        ),
    ],
)
def test_the_cache_is_rebuilt_whenever_what_made_the_vectors_changed(
    tmp_path: Path, paragraphs: tuple[str, ...], embedder: CountingEmbedder, embeds_again: bool
) -> None:
    DenseIndex.build_cached(_inventory(REVENUE, RISK), CountingEmbedder(), tmp_path)
    inventory = _inventory(*paragraphs)

    index = DenseIndex.build_cached(inventory, embedder, tmp_path)

    assert (embedder.embedded > 0) is embeds_again, f"embedded {embedder.embedded} passages"
    fresh = DenseIndex.build(inventory, embedder)
    assert index.ids == fresh.ids
    assert np.array_equal(index.matrix, fresh.matrix), "the cache served vectors for other text"


def _truncate(path: Path) -> None:
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


def _another_inventorys_index(path: Path) -> None:
    DenseIndex.build(_inventory(DIVIDEND), BagOfWordsEmbedder()).save(path)


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(_truncate, id="a truncated file"),
        pytest.param(lambda path: path.write_bytes(b""), id="an empty file"),
        pytest.param(lambda path: path.write_bytes(b"not an index"), id="not an index at all"),
        pytest.param(_another_inventorys_index, id="another inventory's index in its place"),
    ],
)
def test_a_damaged_cache_file_is_rebuilt_rather_than_raised_on(
    tmp_path: Path, damage: Callable[[Path], object]
) -> None:
    inventory = _inventory(REVENUE, RISK)
    embedder = CountingEmbedder()
    DenseIndex.build_cached(inventory, embedder, tmp_path)
    damage(cache_file(inventory, embedder, tmp_path))

    index = DenseIndex.build_cached(inventory, embedder, tmp_path)

    assert embedder.embedded == 2 * len(inventory), "the damaged file was not rebuilt"
    assert index.ids == tuple(inventory.ids())
    assert DenseIndex.load(cache_file(inventory, embedder, tmp_path)).ids == index.ids


def test_the_cache_keeps_one_file_per_strategy(tmp_path: Path) -> None:
    DenseIndex.build_cached(_inventory(REVENUE, RISK), CountingEmbedder(), tmp_path)
    DenseIndex.build_cached(_inventory(REVENUE, DIVIDEND), CountingEmbedder(), tmp_path)

    assert len(list(tmp_path.glob("index-sentence-*.npz"))) == 1, "superseded files were kept"
