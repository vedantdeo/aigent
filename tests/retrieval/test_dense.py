"""The dense index, brute-force cosine: ranking, ties, the two shapes it refuses to build from,
and the copy of it kept on disk.

Runs against `BagOfWordsEmbedder`, so these test the index's arithmetic rather than any model's
quality.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from aigent.retrieval import dense
from aigent.retrieval.chunk import Document, Inventory, by_sentence
from aigent.retrieval.dense import DenseIndex, read_store, store_file, text_key
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


def test_a_second_build_is_served_from_the_store_without_embedding_again(tmp_path: Path) -> None:
    inventory = _inventory(REVENUE, RISK, DIVIDEND)
    embedder = CountingEmbedder()

    first = DenseIndex.build_cached(inventory, embedder, tmp_path)
    second = DenseIndex.build_cached(inventory, embedder, tmp_path)

    assert embedder.embedded == len(inventory), "the second build embedded the corpus again"
    assert second.ids == first.ids
    assert np.array_equal(second.matrix, first.matrix)


@pytest.mark.parametrize(
    ("paragraphs", "embedder", "embeds"),
    [
        pytest.param((REVENUE, RISK), CountingEmbedder(), 0, id="nothing changed"),
        # Same ids, different text: the case a store keyed on ids would serve stale.
        pytest.param((REVENUE, DIVIDEND), CountingEmbedder(), 1, id="a chunk's text changed"),
        pytest.param((REVENUE, RISK, DIVIDEND), CountingEmbedder(), 1, id="a chunk was added"),
        pytest.param((RISK, REVENUE), CountingEmbedder(), 0, id="the same texts reordered"),
        pytest.param(
            (REVENUE, RISK), CountingEmbedder("another-model"), 2, id="another embedding model"
        ),
        pytest.param(
            (REVENUE, RISK), PrefixingEmbedder(), 2, id="the code that embeds a passage changed"
        ),
    ],
)
def test_only_text_the_store_has_not_seen_is_embedded(
    tmp_path: Path, paragraphs: tuple[str, ...], embedder: CountingEmbedder, embeds: int
) -> None:
    DenseIndex.build_cached(_inventory(REVENUE, RISK), CountingEmbedder(), tmp_path)
    inventory = _inventory(*paragraphs)

    index = DenseIndex.build_cached(inventory, embedder, tmp_path)

    assert embedder.embedded == embeds, f"embedded {embedder.embedded} passages"
    fresh = DenseIndex.build(inventory, embedder)
    assert index.ids == fresh.ids
    assert np.array_equal(index.matrix, fresh.matrix), "a row holds another text's vector"


def test_indexes_that_share_text_share_its_vectors(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    DenseIndex.build_cached(_inventory(REVENUE, RISK), embedder, tmp_path)

    other = DenseIndex.build_cached(
        Inventory.build("other", [Document("x", RISK + DIVIDEND)], by_sentence(400, 0)),
        embedder,
        tmp_path,
    )

    assert len(other) == 1 and embedder.embedded == 3, "a new text, whatever the index"
    DenseIndex.build_cached(_inventory(DIVIDEND, RISK), embedder, tmp_path)
    assert embedder.embedded == 4, "only DIVIDEND alone was new"


def test_a_text_an_inventory_holds_twice_is_embedded_once(tmp_path: Path) -> None:
    embedder = CountingEmbedder()

    index = DenseIndex.build_cached(_inventory(REVENUE, REVENUE), embedder, tmp_path)

    assert embedder.embedded == 1
    assert len(index) == 2 and np.array_equal(index.matrix[0], index.matrix[1])


def _truncate(path: Path) -> None:
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


def _misaligned(path: Path) -> None:
    np.savez(path, keys=np.array(["a", "b"]), matrix=np.eye(4, dtype=np.float32)[:1])


def _a_key_twice(path: Path) -> None:
    np.savez(path, keys=np.array(["a", "a"]), matrix=np.eye(4, dtype=np.float32)[:2])


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(_truncate, id="a truncated file"),
        pytest.param(lambda path: path.write_bytes(b""), id="an empty file"),
        pytest.param(lambda path: path.write_bytes(b"not a store"), id="not a store at all"),
        pytest.param(_misaligned, id="keys and vectors that disagree"),
        pytest.param(_a_key_twice, id="a key twice"),
    ],
)
def test_a_damaged_store_is_started_afresh_rather_than_raised_on(
    tmp_path: Path, damage: Callable[[Path], object]
) -> None:
    inventory = _inventory(REVENUE, RISK)
    embedder = CountingEmbedder()
    DenseIndex.build_cached(inventory, embedder, tmp_path)
    damage(store_file(embedder, tmp_path))

    index = DenseIndex.build_cached(inventory, embedder, tmp_path)

    assert embedder.embedded == 2 * len(inventory), "the damaged store was not rebuilt"
    assert index.ids == tuple(inventory.ids())
    keys, _ = read_store(store_file(embedder, tmp_path))
    assert sorted(keys) == sorted(text_key(t) for t in inventory.texts())


def test_the_store_keeps_one_file_per_model_written_whole(tmp_path: Path) -> None:
    DenseIndex.build_cached(_inventory(REVENUE, RISK), CountingEmbedder(), tmp_path)
    DenseIndex.build_cached(_inventory(REVENUE), PrefixingEmbedder(), tmp_path)
    DenseIndex.build_cached(_inventory(REVENUE), CountingEmbedder("another/model"), tmp_path)

    names = sorted(path.name for path in tmp_path.iterdir())
    assert len(names) == 2, f"one store per model, its older version replaced: {names}"
    assert [n.split("-")[1] for n in names] == ["another_model", "bag"], names
    assert not list(tmp_path.glob("*.partial")), "a partial file was left"


def test_building_embeds_in_slices_logging_each_and_matches_one_pass(
    embedder: BagOfWordsEmbedder, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    inventory = _inventory(REVENUE, RISK, DIVIDEND, REVENUE + RISK, RISK + DIVIDEND)
    monkeypatch.setattr(dense, "EMBED_LOG_EVERY", 2)

    with caplog.at_level(logging.INFO, logger="aigent.retrieval.dense"):
        index = DenseIndex.build(inventory, embedder)

    assert np.allclose(index.matrix, embedder.embed_documents(inventory.texts()))
    lines = [r.getMessage() for r in caplog.records]
    assert [line.split(" with ")[0] for line in lines] == [
        "sentence: embedded 2/5 (40%)",
        "sentence: embedded 4/5 (80%)",
        "sentence: embedded 5/5 (100%)",
    ], lines


def test_a_cached_build_says_whether_it_embedded_or_read(
    embedder: BagOfWordsEmbedder, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    inventory = _inventory(REVENUE, RISK)

    with caplog.at_level(logging.INFO, logger="aigent.retrieval.dense"):
        DenseIndex.build_cached(inventory, embedder, tmp_path)
        DenseIndex.build_cached(inventory, embedder, tmp_path)

    said = [r.getMessage() for r in caplog.records]
    assert said[0] == "sentence: 0 of 2 texts already embedded with bag-of-words-fake; embedding 2"
    assert said[-1] == "sentence: 2 of 2 texts already embedded with bag-of-words-fake; embedding 0"


def test_texts_differing_by_a_character_never_share_a_key() -> None:
    texts = [f"Revenue for FY{n % 100:02d} was ₹ {n:,} crore." for n in range(5000)]

    keys = {text_key(text) for text in texts}

    assert len(keys) == len(texts), f"{len(texts) - len(keys)} texts would share another's vector"
    assert text_key("a") != text_key("a ") and text_key("a") == text_key("a")
