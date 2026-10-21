"""The retrieval eval's own logic: a label's quote resolved to chunk ids, the two ranking tasks,
and the note naming labels no chunk holds. No model is loaded; indexes and rankers are fakes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import JsonValue

from aigent.config import SEARCH_METHOD
from aigent.evals.dataset import Case, DatasetError, digest
from aigent.evals.runner import EvalRun
from aigent.retrieval import evaluate
from aigent.retrieval.chunk import Chunk, Document, Inventory
from aigent.retrieval.dense import DenseIndex
from aigent.retrieval.embed import Embedder, Vectors
from aigent.retrieval.evaluate import (
    compare_chunking,
    compare_retrieval,
    label_of,
    load_cases,
    ranker_task,
    report_unresolved,
    resolve,
    retrieval_task,
)
from aigent.retrieval.hits import Hit
from aigent.retrieval.rank import METHODS, Ranker

from ..conftest import BagOfWordsEmbedder, KeywordReranker

Compare = Callable[
    [Sequence[Case], Sequence[Document], Embedder, int, Sequence[Path], int, bool], EvalRun
]


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
        self.asked: list[tuple[str, int, object]] = []

    def rank(self, query: str, k: int, doc_id: object = None) -> list[Hit]:
        self.asked.append((query, k, doc_id))
        return [Hit("ITC-FY25#0002", 3.0)]


SCOPES = [
    pytest.param(["ITC-FY25", "lookup"], "ITC-FY25", id="a case naming one report"),
    pytest.param(["lookup"], None, id="a case naming no report"),
    pytest.param(["ITC-FY25", "RELIANCE-FY25"], None, id="a case naming two reports"),
]


def _tagged(tags: list[str]) -> Case:
    return Case(id="q1", input={"question": "?"}, expected={}, tags=tags)


@pytest.mark.parametrize(("tags", "report"), SCOPES)
def test_the_dense_task_ranks_only_the_report_its_case_names(
    tags: list[str], report: str | None
) -> None:
    queries = cast(dict[str, Vectors], {"q1": np.ones(4, dtype=np.float32)})
    task = retrieval_task(cast(DenseIndex, _Index()), queries, INVENTORY, k=2)

    outcome = task(_tagged(tags))

    if report is None:
        assert outcome.error == "case q1 names no single report to search", outcome
    else:
        assert outcome.error is None and outcome.usage is None, "ranking spends nothing"
        assert outcome.output == {"retrieved": ["ITC-FY25#0002"], "scores": [0.9]}, outcome


def test_the_dense_task_reports_a_case_with_no_query_vector() -> None:
    task = retrieval_task(cast(DenseIndex, _Index()), {}, INVENTORY, k=2)

    assert task(_tagged(["ITC-FY25"])).error == "case q1 has no query vector"


@pytest.mark.parametrize(("tags", "report"), SCOPES)
def test_a_ranker_task_ranks_the_question_within_its_report(
    tags: list[str], report: str | None
) -> None:
    ranker = _Ranker()

    outcome = ranker_task(cast(Ranker, ranker), k=3)(_tagged(tags))

    if report is None:
        assert ranker.asked == [], "an unscoped case is never searched across every report"
        assert outcome.error == "case q1 names no single report to search", outcome
    else:
        assert ranker.asked == [("?", 3, report)]
        assert outcome.output == {"retrieved": ["ITC-FY25#0002"], "scores": [3.0]}, outcome


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


def _dataset(path: Path, *ids: str) -> Path:
    rows = (
        f'{{"id": "{i}", "input": {{"question": "?"}}, "expected": {{"quote": "q"}}, '
        f'"tags": ["ITC-FY25"]}}'
        for i in ids
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_two_files_load_as_one_set_in_file_order(tmp_path: Path) -> None:
    prose = _dataset(tmp_path / "prose.jsonl", "rq-002", "rq-001")
    tables = _dataset(tmp_path / "tables.jsonl", "rt-001")

    assert [case.id for case in load_cases([prose, tables])] == ["rq-002", "rq-001", "rt-001"]


def test_an_id_two_files_share_is_refused(tmp_path: Path) -> None:
    prose = _dataset(tmp_path / "prose.jsonl", "rq-001")
    tables = _dataset(tmp_path / "tables.jsonl", "rt-001", "rq-001")

    with pytest.raises(DatasetError, match="'rq-001' is also in prose.jsonl"):
        load_cases([prose, tables])


def test_one_file_is_labelled_as_before(tmp_path: Path) -> None:
    prose = _dataset(tmp_path / "prose.jsonl", "rq-001")

    assert label_of([prose]) == ("prose.jsonl", digest(prose))


@pytest.mark.parametrize("edited", [0, 1], ids=["the first file edited", "the second file edited"])
def test_the_joined_digest_moves_with_either_file(tmp_path: Path, edited: int) -> None:
    paths = [_dataset(tmp_path / "prose.jsonl", "rq-001"), _dataset(tmp_path / "t.jsonl", "rt-1")]
    name, before = label_of(paths)

    _dataset(paths[edited], "rx-999")

    assert name == "prose.jsonl+t.jsonl"
    assert label_of(paths)[1] != before, "a report must say the data changed"


@pytest.mark.parametrize(
    "compare", [compare_chunking, compare_retrieval], ids=["chunking", "retrieval methods"]
)
def test_a_comparison_over_two_files_reports_them_as_one_set(
    compare: Compare, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "LocalReranker", KeywordReranker)
    paths = [_dataset(tmp_path / "prose.jsonl", "rq-001"), _dataset(tmp_path / "t.jsonl", "rt-1")]
    documents = [Document(doc_id="ITC-FY25", text="Revenue q grew. Dividend q was paid.")]

    run = compare(load_cases(paths), documents, BagOfWordsEmbedder(), 5, paths, 30, False)

    assert (run.dataset, run.digest) == label_of(paths)
    assert {row.case_id for row in run.rows} == {"rq-001", "rt-1"}, "both files are scored"
    assert all(row.error is None for row in run.rows), [row.error for row in run.rows]


def test_the_methods_comparison_scores_the_keyed_search_beside_the_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "LocalReranker", KeywordReranker)
    paths = [_dataset(tmp_path / "prose.jsonl", "rq-001")]
    documents = [Document(doc_id="ITC-FY25", text="Revenue q grew. Dividend q was paid.")]

    run = compare_retrieval(load_cases(paths), documents, BagOfWordsEmbedder(), 5, paths, 30, False)

    assert run.variants == (*METHODS, f"keys+{SEARCH_METHOD}"), run.variants
    assert all(row.error is None for row in run.rows), [row.error for row in run.rows]
