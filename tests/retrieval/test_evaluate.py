"""The retrieval eval's own logic: a label's quote resolved to chunk ids, the two ranking tasks,
and the note naming labels no chunk holds. No model is loaded; indexes and rankers are fakes."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pydantic import JsonValue

from aigent.evals.dataset import Case
from aigent.retrieval.chunk import Chunk, Inventory
from aigent.retrieval.dense import DenseIndex
from aigent.retrieval.embed import Vectors
from aigent.retrieval.evaluate import ranker_task, report_unresolved, resolve, retrieval_task
from aigent.retrieval.hits import Hit
from aigent.retrieval.rank import Ranker


def _chunk(id_: str, text: str) -> Chunk:
    return Chunk(id=id_, text=text, doc_id=id_.split("#")[0], ordinal=0, start=0)


CHUNKS = (
    _chunk("ITC-FY25#0001", "Cigarettes net segment revenue grew 6.5 per cent."),
    _chunk("ITC-FY25#0002", "The Board recommended a final dividend of Rs 7.85 per share."),
    _chunk("RIL-FY25#0001", "The Board recommended a final dividend of Rs 5.50 per share."),
)
INVENTORY = Inventory("toy", CHUNKS, {chunk.id: chunk for chunk in CHUNKS})


def _case(quote: object = None, **expected: JsonValue) -> Case:
    label: dict[str, JsonValue] = {**expected}
    if quote is not None:
        label["quote"] = cast(JsonValue, quote)
    return Case(id="q1", input={"question": "?"}, expected=label, tags=["lookup"])


@pytest.mark.parametrize(
    ("quote", "relevant"),
    [
        pytest.param("grew 6.5 per cent", ["ITC-FY25#0001"], id="a quote one chunk holds"),
        pytest.param(
            "final dividend of", ["ITC-FY25#0002", "RIL-FY25#0001"], id="a quote two chunks hold"
        ),
        pytest.param("6.5 per cent. The Board", [], id="a quote straddling two chunks"),
        pytest.param(None, [], id="a case with no quote"),
        pytest.param(42, [], id="a quote that is not text"),
    ],
)
def test_a_label_resolves_to_every_chunk_holding_its_quote(
    quote: object, relevant: list[str]
) -> None:
    [resolved] = resolve([_case(quote, answer="x")], INVENTORY)

    assert resolved.expected["relevant"] == relevant, resolved.expected
    assert resolved.expected["answer"] == "x", "the rest of the label is kept"
    assert (resolved.id, resolved.input, resolved.tags) == ("q1", {"question": "?"}, ["lookup"])


def test_a_label_no_chunk_holds_stays_in_the_set() -> None:
    cases = [_case("grew 6.5"), _case("nowhere in the corpus"), _case("Rs 5.50")]

    resolved = resolve(cases, INVENTORY)

    assert [case.expected["relevant"] for case in resolved] == [
        ["ITC-FY25#0001"],
        [],
        ["RIL-FY25#0001"],
    ], "dropping the unresolved row would raise this strategy's score"


class _Index:
    """Answers any vector with the same two hits."""

    def search(self, vector: Vectors, k: int) -> list[Hit]:
        del vector
        return [Hit("ITC-FY25#0002", 0.9), Hit("RIL-FY25#0001", 0.7)][:k]


class _Ranker:
    def __init__(self) -> None:
        self.asked: list[tuple[str, int]] = []

    def rank(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]:
        del doc_id
        self.asked.append((query, k))
        return [Hit("RIL-FY25#0001", 3.0)]


def test_the_dense_task_ranks_a_pre_embedded_question() -> None:
    queries = {"q1": np.ones(4, dtype=np.float32)}
    task = retrieval_task(cast(DenseIndex, _Index()), cast(dict[str, Vectors], queries), k=2)

    found = task(_case("x"))
    missing = task(Case(id="q9", input={"question": "?"}))

    assert found.error is None and found.usage is None, "ranking spends nothing"
    assert found.output == {"retrieved": ["ITC-FY25#0002", "RIL-FY25#0001"], "scores": [0.9, 0.7]}
    assert missing.error == "case q9 has no query vector"


def test_a_ranker_task_ranks_the_question_text() -> None:
    ranker = _Ranker()

    outcome = ranker_task(cast(Ranker, ranker), k=3)(_case("x"))

    assert ranker.asked == [("?", 3)]
    assert outcome.output == {"retrieved": ["RIL-FY25#0001"], "scores": [3.0]}


@pytest.mark.parametrize(
    ("relevant", "printed"),
    [
        pytest.param([["a#1"], ["b#1"]], "", id="every label resolved: nothing to say"),
        pytest.param([["a#1"], []], "1 labels straddle a boundary under toy: q1", id="one lost"),
    ],
)
def test_unresolved_labels_are_named(
    relevant: list[list[str]], printed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = [
        Case(id=f"q{i}", input={}, expected={"relevant": cast(JsonValue, ids)})
        for i, ids in enumerate(relevant)
    ]

    report_unresolved("toy", cases)

    assert capsys.readouterr().out.strip() == printed
