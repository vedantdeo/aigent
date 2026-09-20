"""The LangChain adapter, without LangChain's models and without its PDFs.

Skipped entirely when the `compare` group is not installed, which is the case in CI — the point of
putting the framework in a non-default group is that nothing else has to carry it.

The load-bearing test is the last one. The first real run of this module scored 0.000 on every
metric with no error raised, because retrieved chunks were matched back to labels by `id(chunk)`
and `InMemoryVectorStore` hands back reconstructed objects. Empty id lists grade as clean misses,
so a broken mapping is indistinguishable from a bad retriever unless something refuses to be quiet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core", reason="needs `uv sync --group compare`")

from langchain_core.documents import Document as LcDocument  # noqa: E402

from entropic.evals.dataset import Case  # noqa: E402
from entropic.retrieval.langchain_rag import langchain_task, prepare  # noqa: E402


class _Retriever:
    """Stands in for a `VectorStoreRetriever`: returns what it was given, in order."""

    def __init__(self, found: list[LcDocument]) -> None:
        self._found = found

    def invoke(self, query: str) -> list[LcDocument]:
        del query
        return self._found


def _chunks() -> list[LcDocument]:
    return [
        LcDocument(
            page_content=f"passage {i} about dividends",
            metadata={"source": "/corpus/ITC-FY25.pdf", "page": i, "start_index": i * 100},
        )
        for i in range(3)
    ]


def _case() -> Case:
    return Case.model_validate({"id": "rq-001", "input": {"question": "what dividend?"}})


def test_the_inventory_and_the_metadata_carry_the_identical_id() -> None:
    """The one property that keeps this comparison honest. The inventory is what a labelled quote
    resolves against; the metadata is what a retrieved chunk is recognised by. Derive those from
    two formulas and they drift apart into labels nothing can match — which reads as a retriever
    that missed everything, not as a broken adapter."""
    chunks = _chunks()

    inventory = prepare(chunks)

    assert [c.metadata["chunk_id"] for c in chunks] == [c.id for c in inventory.chunks]
    assert [c.id for c in inventory.chunks] == [
        "ITC-FY25#0000",
        "ITC-FY25#0001",
        "ITC-FY25#0002",
    ]


def test_the_inventory_renumbers_pages_from_one() -> None:
    """LangChain's loader 0-indexes pages; every citation in this repo is 1-indexed. Off by one in
    the one field a reader would use to check a quote."""
    inventory = prepare(_chunks())

    assert [chunk.page for chunk in inventory.chunks] == [1, 2, 3]


def test_a_retrieved_chunk_is_reported_by_the_id_in_its_metadata() -> None:
    chunks = _chunks()
    prepare(chunks)

    outcome = langchain_task(_Retriever([chunks[2], chunks[0]]))(_case())

    assert outcome.error is None
    assert outcome.output["retrieved"] == ["ITC-FY25#0002", "ITC-FY25#0000"], outcome.output
    assert outcome.usage is None, "retrieval spends nothing"


def test_a_chunk_with_no_id_is_an_error_row_rather_than_a_quiet_miss() -> None:
    """The regression test for a run that reported 0.000 across every metric and raised nothing.
    Dropping an unmappable chunk makes a broken adapter look exactly like a bad retriever."""
    unstamped = _chunks()  # deliberately not passed through `prepare`

    outcome = langchain_task(_Retriever(unstamped))(_case())

    assert outcome.error is not None and "chunk_id" in outcome.error, outcome
    assert not outcome.output, "a row that cannot be mapped must not be graded as a miss"
