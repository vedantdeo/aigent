"""BM25: what it ranks, and the two things it must do that a dense retriever cannot.

The load-bearing tests are the rare-token one and the no-match one. The first is the entire reason
this module exists beside the vector store — an acronym or a figure that an embedding smears is
exactly what an inverted index finds. The second is what keeps a fused ranking honest: a chunk
sharing no word with the query is not a weak match, and padding it in would put text no ranker
endorsed in front of a reader.
"""

from __future__ import annotations

import pytest

from entropic.retrieval.chunk import Document, Inventory, by_sentence
from entropic.retrieval.sparse import Bm25Index, tokenise
from entropic.retrieval.store import rank_ids

SACE = (
    "The company raised green SACE Push facilities supported by the Italian Export Credit Agency "
    "during the year under review. "
)
DIVIDEND = (
    "The board recommended a final dividend of ten rupees per equity share for the year ended "
    "March 2025. "
)
GENERIC = (
    "The company continued to review its financing arrangements during the year under review as "
    "part of its normal course of business. "
)


def _index(*texts: str) -> Bm25Index:
    documents = [Document(doc_id=f"doc-{i}", text=t) for i, t in enumerate(texts, start=1)]
    return Bm25Index.build(Inventory.build("sentence", documents, by_sentence(400, 0)))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Revenue rose 12%", ["revenue", "rose", "12"], id="a bare percentage"),
        # The whole argument for a lexical ranker: an embedding cannot keep these digits apart.
        pytest.param("Rs 34,000 crores", ["rs", "34,000", "crores"], id="a grouped figure"),
        pytest.param(
            "margin of 52.3 per cent",
            ["margin", "of", "52.3", "per", "cent"],
            id="a decimal survives whole",
        ),
        pytest.param("KPMG's BRSR report", ["kpmg", "s", "brsr", "report"], id="case folds"),
        pytest.param("", [], id="nothing at all"),
    ],
)
def test_tokenise_keeps_a_number_in_one_piece(text: str, expected: list[str]) -> None:
    assert tokenise(text) == expected


def test_a_rare_token_outranks_a_chunk_that_merely_shares_common_words() -> None:
    """`SACE` appears once; `the`, `company`, `year` and `review` appear in both chunks. idf is
    what makes the rare term decide, and it is why this finds what a dense retriever missed."""
    index = _index(SACE, GENERIC)

    hits = index.search("Which export credit agency supported the green financing facilities?", k=2)

    assert rank_ids(hits)[0] == "doc-1#0000", [(h.chunk_id, round(h.score, 2)) for h in hits]


def test_a_chunk_sharing_no_query_term_is_absent_rather_than_ranked_last() -> None:
    index = _index(SACE, DIVIDEND)

    hits = index.search("dividend", k=5)

    assert rank_ids(hits) == ["doc-2#0000"], "only the chunk holding the term comes back"


def test_a_query_of_unknown_words_returns_nothing_rather_than_an_arbitrary_order() -> None:
    index = _index(SACE, DIVIDEND)

    assert index.search("cryptocurrency staking derivatives", k=5) == []


def test_scoring_touches_only_the_chunks_that_share_a_term() -> None:
    """`score` is sparse on purpose — an absent row is "no match", which is how `search` tells
    that apart from "matched and scored badly" without needing a threshold."""
    index = _index(SACE, DIVIDEND, GENERIC)

    scored = index.score("dividend")

    assert len(scored) == 1, scored
    assert all(value > 0 for value in scored.values())


def test_rank_all_returns_the_whole_ranking_because_fusion_reads_ranks() -> None:
    """Truncating here would hide a chunk that is rank 8 lexically and rank 3 densely — exactly
    the chunk fusion exists to promote."""
    index = _index(SACE, DIVIDEND, GENERIC)

    # "company" and "review" are in SACE and GENERIC; DIVIDEND has neither, so it must be absent.
    assert len(index.rank_all("company review")) == 2, "the two chunks sharing those words"


def test_an_empty_corpus_scores_nothing_rather_than_dividing_by_zero() -> None:
    index = Bm25Index.build(Inventory.build("sentence", [], by_sentence(400, 0)))

    assert len(index) == 0
    assert index.search("anything", k=3) == []


def test_a_k_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _index(SACE).search("anything", k=0)
