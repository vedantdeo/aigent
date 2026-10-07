"""The question generator's free half: what gets asked about, and what gets thrown away.

`verify` is the load-bearing one. A quote the model paraphrased resolves to no chunk under any
strategy, so it reads as four retrieval failures rather than as one bad label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aigent.evals.dataset import Case, load_jsonl
from aigent.retrieval.chunk import Chunk, Document, Inventory, squeeze
from aigent.retrieval.questions import (
    DATASET,
    Question,
    eligible,
    sample_passages,
    to_row,
    verify,
    write_dataset,
)

PROSE = (
    "The Board of Directors has recommended a final dividend of Rs. 10 per equity share for the "
    "financial year ended 31st March 2025. Together with the interim dividend already paid, this "
    "takes the total distribution for the year to Rs. 16.50 per equity share, which represents a "
    "payout ratio broadly consistent with the policy adopted by the Board in earlier years. The "
    "record date for the purpose of payment will be notified to the stock exchanges in due course. "
)
# Long enough to clear the length floor on its own, so the row tests the prose rule and not that.
TABLE = (
    "12,345 6,789 1,024 88.4 91.2 7,654 3,210 55.5 66.6 4,321 9,876 2,468 13.7 24.9 8,642 1,357 "
)


def _chunk(text: str, doc_id: str = "RIL-FY25", ordinal: int = 0) -> Chunk:
    return Chunk(id=f"{doc_id}#{ordinal:04d}", text=text, doc_id=doc_id, ordinal=ordinal, start=0)


@pytest.mark.parametrize(
    ("text", "usable"),
    [
        pytest.param(PROSE, True, id="prose long enough to hold a fact"),
        pytest.param(PROSE[:200], False, id="prose too short to be worth a question"),
        pytest.param(TABLE * 6, False, id="a table, where a question would be about formatting"),
    ],
)
def test_only_passages_worth_asking_about_are_eligible(text: str, usable: bool) -> None:
    assert eligible(_chunk(text)) is usable


def test_the_sample_is_spread_and_identical_between_runs() -> None:
    from aigent.retrieval.chunk import Inventory

    document = Document(doc_id="RIL-FY25", text=PROSE * 40)
    inventory = Inventory.build(
        "sentence", [document], lambda doc: [_chunk(PROSE, ordinal=i) for i in range(40)]
    )

    first = sample_passages(inventory, 8)
    again = sample_passages(inventory, 8)

    assert len(first) == 8
    assert [c.id for c in first] == [c.id for c in again], "a moving sample cannot be compared"
    assert first[0] is inventory.chunks[0], "the spread starts at the beginning"
    assert first[-1] is inventory.chunks[-1], "and ends at the end"
    assert len({c.id for c in first}) == 8, "no passage asked about twice"


@pytest.mark.parametrize(
    ("n", "taken"),
    [
        pytest.param(0, [], id="none asked for"),
        pytest.param(-2, [], id="a negative count"),
        pytest.param(1, [0], id="one: the first eligible passage"),
        pytest.param(3, [0, 2, 4], id="three spread across five, ends included"),
    ],
)
def test_the_sample_takes_evenly_spaced_passages(n: int, taken: list[int]) -> None:
    chunks = tuple(_chunk(PROSE, ordinal=i) for i in range(5))
    inventory = Inventory("sentence", chunks, {chunk.id: chunk for chunk in chunks})

    assert [chunk.ordinal for chunk in sample_passages(inventory, n)] == taken


def test_asking_for_more_passages_than_exist_returns_what_there_is() -> None:
    from aigent.retrieval.chunk import Inventory

    inventory = Inventory.build(
        "sentence",
        [Document(doc_id="RIL-FY25", text=PROSE * 3)],
        lambda doc: [_chunk(PROSE, ordinal=i) for i in range(3)],
    )

    assert len(sample_passages(inventory, 50)) == 3


@pytest.mark.parametrize(
    ("question", "quote", "usable", "problem"),
    [
        pytest.param(
            "What final dividend did Reliance recommend?",
            "recommended a final dividend of Rs. 10 per equity share",
            True,
            None,
            id="a verbatim quote from the document",
        ),
        pytest.param(
            "What final dividend did Reliance recommend?",
            "recommended   a FINAL dividend of Rs. 10 per equity share",
            True,
            None,
            id="whitespace and case are forgiven",
        ),
        pytest.param(
            "What final dividend did Reliance recommend?",
            "the board proposed a final dividend of ten rupees a share",
            True,
            "verbatim",
            id="a paraphrase is caught here, not four failures later",
        ),
        pytest.param(
            "What final dividend did Reliance recommend?",
            "Rs. 10",
            True,
            "too short",
            id="a quote too short to identify one passage",
        ),
        pytest.param("", "", True, "empty", id="an empty question"),
        pytest.param(
            "q",
            "a quote long enough to pass the floor",
            False,
            "unusable",
            id="the model said the passage was not worth a question",
        ),
        pytest.param(
            "Which Board recommended a final dividend of Rs. 10 per equity share?",
            "recommended a final dividend of Rs. 10 per equity share",
            True,
            "contains its own answer",
            id="a question that answers itself measures nothing",
        ),
    ],
)
def test_a_question_is_only_kept_when_its_quote_is_really_in_the_document(
    question: str, quote: str, usable: bool, problem: str | None
) -> None:
    document = Document(doc_id="RIL-FY25", text=PROSE)
    candidate = Question(usable=usable, question=question, quote=quote, topic="dividend")

    verdict = verify(candidate, document)

    if problem is None:
        assert verdict is None, verdict
    else:
        assert verdict is not None and problem in verdict, verdict


def test_a_quote_two_reports_share_is_not_a_label_for_either_of_them() -> None:
    """A boilerplate sentence identifies no document, so it resolves to chunks in both and the
    retriever is marked wrong for returning the one the question was not about."""
    shared = "Audited by the independent auditors appointed at the annual general meeting."
    candidate = Question(
        usable=True, question="Who audits Reliance?", quote=shared, topic="assurance"
    )

    verdict = verify(
        candidate,
        Document(doc_id="RIL-FY25", text=PROSE + shared),
        [Document(doc_id="ITC-FY25", text=shared + PROSE)],
    )

    assert verdict is not None and "ITC-FY25" in verdict, verdict
    assert verify(candidate, Document(doc_id="RIL-FY25", text=PROSE + shared)) is None


def test_a_row_carries_the_document_and_the_topic_as_tags() -> None:
    candidate = Question(
        usable=True, question="What dividend?", quote="a quote", topic="  Dividend  "
    )

    row = to_row(7, _chunk(PROSE, doc_id="ITC-FY25"), candidate)

    assert row["id"] == "rq-007"
    assert row["input"] == {"question": "What dividend?"}
    assert row["expected"] == {"quote": "a quote"}
    assert row["tags"] == ["ITC-FY25", "dividend"]


def test_the_dataset_is_written_in_id_order_one_row_per_line(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {"id": "rq-003", "input": {"question": "c"}, "expected": {"quote": "z"}, "tags": []},
        {"id": "rq-001", "input": {"question": "a"}, "expected": {"quote": "x"}, "tags": []},
    ]

    path = write_dataset(rows, tmp_path / "retrieval.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()

    assert [json.loads(line)["id"] for line in lines] == ["rq-001", "rq-003"]


# --- The dataset as an asset ------------------------------------------------------------------
# Corpus-free on purpose: CI has no PDFs, so these check the shape of the labels rather than
# whether they resolve. Resolution is what `resolvable` reports at run time, against a corpus.

CASES = load_jsonl(DATASET)


def test_the_question_set_is_the_size_and_spread_it_claims() -> None:
    per_document = {
        tag: sum(1 for c in CASES if c.tags[0] == tag) for c in CASES for tag in [c.tags[0]]
    }

    assert len(CASES) >= 50, "the roadmap asks for 50 questions"
    assert len(per_document) == 3, per_document
    assert min(per_document.values()) >= 10, f"one report is barely represented: {per_document}"
    assert len({c.id for c in CASES}) == len(CASES), "ids must be unique"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_every_row_is_a_usable_retrieval_label(case: Case) -> None:
    """What a row has to be before it can score anything, checked per row so the id names it."""
    question = case.input.get("question")
    quote = case.expected.get("quote")

    assert isinstance(question, str) and question.strip(), "no question"
    assert isinstance(quote, str) and quote.strip(), "no quote"
    assert len(squeeze(quote)) >= 25, f"quote too short to identify a passage: {quote!r}"
    assert squeeze(quote) not in squeeze(question), "the question contains its own answer"
    assert len(case.tags) == 2, f"expected [doc_id, topic], got {case.tags}"
    assert case.tags[0].endswith("-FY25"), f"first tag should name the document: {case.tags[0]}"
    assert case.tags[1] == case.tags[1].casefold().strip(), "topic tags are lowercase and trimmed"


def test_no_question_is_asked_twice() -> None:
    asked = [squeeze(str(case.input["question"])) for case in CASES]
    repeated = sorted({q for q in asked if asked.count(q) > 1})

    assert not repeated, f"duplicate questions measure the same thing twice: {repeated}"
