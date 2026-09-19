"""The generation half, for nothing: a scripted client, a fake embedder, no network.

The load-bearing tests are the two about what leaves the module — that every passage reaches the
model tagged with the id it must cite, and that the judge is handed the answer without those ids
beside it. A citation nobody can check against the label is the failure this eval exists to catch,
and it would pass every other test in this file.
"""

from __future__ import annotations

from typing import cast

import anthropic
import pytest
from anthropic.types import MessageTokensCount, Usage

from entropic.config import MAX_TOKENS_ANSWER, MODEL, THINKING_EVAL_PARAM
from entropic.evals.dataset import Case
from entropic.evals.runner import Task
from entropic.retrieval.answer import (
    ARMS,
    Answer,
    answer_task,
    context_block,
    spread,
)
from entropic.retrieval.chunk import Document, Inventory, by_sentence
from entropic.retrieval.store import VectorStore

from ..conftest import BagOfWordsEmbedder

REVENUE = "Revenue for the year rose seven per cent on stronger retail volumes across the country. "
DIVIDEND = "The board recommended a final dividend of ten rupees per equity share for the year. "

ANSWERED = Answer(answered=True, answer="Ten rupees per equity share.", cited=["doc-2#0000"])
DECLINED = Answer(answered=False, answer="", cited=[])


class _Parsed:
    def __init__(self, record: Answer | None) -> None:
        self.parsed_output = record
        self.usage = Usage(input_tokens=1800, output_tokens=40)
        self.stop_reason = "max_tokens" if record is None else "end_turn"


class _Messages:
    def __init__(self, record: Answer | None) -> None:
        self._record = record
        self.sent: list[str] = []
        self.systems: list[object] = []
        self.thinking: list[object] = []
        self.counted = 0

    def count_tokens(self, **_: object) -> MessageTokensCount:
        self.counted += 1
        return MessageTokensCount(input_tokens=1800)

    def parse(self, **kwargs: object) -> _Parsed:
        messages = cast(list[dict[str, str]], kwargs["messages"])
        self.sent.append(messages[0]["content"])
        self.systems.append(kwargs["system"])
        self.thinking.append(kwargs["thinking"])
        assert kwargs["max_tokens"] == MAX_TOKENS_ANSWER
        return _Parsed(self._record)


class _Client:
    def __init__(self, record: Answer | None) -> None:
        self.messages = _Messages(record)


def _inventory() -> Inventory:
    documents = [
        Document(doc_id=f"doc-{i}", text=text)
        for i, text in enumerate([REVENUE, DIVIDEND], start=1)
    ]
    return Inventory.build("sentence", documents, by_sentence(400, overlap_sentences=0))


def _case(question: str = "What dividend did the board recommend?") -> Case:
    return Case.model_validate({"id": "rq-001", "input": {"question": question}})


def _task_and_log(
    arm_name: str = "rag", record: Answer | None = ANSWERED, k: int = 2
) -> tuple[Task, _Messages]:
    inventory = _inventory()
    embedder = BagOfWordsEmbedder()
    store = VectorStore.build(inventory, embedder)
    queries = {"rq-001": embedder.embed_query(str(_case().input["question"]))}
    fake = _Client(record)
    task = answer_task(
        ARMS[arm_name],
        inventory,
        store,
        queries,
        k=k,
        client=cast(anthropic.Anthropic, fake),
    )
    return task, fake.messages


def test_every_passage_reaches_the_model_tagged_with_the_id_it_must_cite() -> None:
    """A citation the label cannot be checked against is worth nothing, and `cites_relevant` reads
    the ids straight out of the answer — so the ids have to be in the prompt to begin with."""
    inventory = _inventory()

    block = context_block(list(inventory.chunks))

    for chunk in inventory.chunks:
        assert f'id="{chunk.id}"' in block, chunk.id
        assert chunk.text in block


def test_the_closed_book_arm_sends_the_question_and_no_passages() -> None:
    task, log = _task_and_log(arm_name="closed_book")

    task(_case())

    assert log.sent == ["question: What dividend did the board recommend?"]
    assert "<passage" not in log.sent[0], "the arm that measures the gap must not be given context"


def test_the_rag_arm_sends_the_retrieved_passages_ahead_of_the_question() -> None:
    task, log = _task_and_log(k=2)

    task(_case())

    sent = log.sent[0]
    assert sent.count("<passage") == 2, sent
    assert sent.index("<passage") < sent.index("question:"), "context first, then the question"


def test_a_parsed_answer_becomes_a_gradeable_outcome() -> None:
    task, log = _task_and_log()

    outcome = task(_case())

    assert outcome.error is None
    assert outcome.output["answer"] == "Ten rupees per equity share."
    assert outcome.output["cited"] == ["doc-2#0000"], "what `cites_relevant` grades"
    assert outcome.usage is not None and outcome.model == MODEL, "the row can be billed"
    assert log.counted == 1, "the paid call was counted first"


def test_the_judge_is_handed_the_answer_without_the_ids_beside_it() -> None:
    """`LlmJudge` reads `raw` when it is set. Chunk ids in the judged text are noise at best and a
    hint at worst — the judge grades the fact against the reference, not the retrieval."""
    task, _ = _task_and_log()

    outcome = task(_case())

    assert outcome.raw == "Ten rupees per equity share."
    assert "doc-2#0000" not in (outcome.raw or "")


def test_a_declined_answer_says_so_rather_than_reaching_the_judge_empty() -> None:
    """Declining is the right behaviour with no passages and still not a correct answer. An empty
    string would reach the judge as a missing field rather than as a refusal."""
    task, _ = _task_and_log(record=DECLINED)

    outcome = task(_case())

    assert outcome.output["answered"] is False
    assert outcome.raw == "(declined to answer)"
    assert outcome.output["cited"] == []


def test_a_refusal_to_parse_is_an_error_row_that_still_bills() -> None:
    task, _ = _task_and_log(record=None)

    outcome = task(_case())

    assert outcome.error is not None and "max_tokens" in outcome.error
    assert outcome.usage is not None, "it still cost money, so the runner must still see usage"
    assert outcome.output == {}


def test_an_eval_run_turns_extended_thinking_off() -> None:
    task, log = _task_and_log()

    task(_case())

    assert log.thinking == [THINKING_EVAL_PARAM], "an eval is a measurement, not a conversation"


def test_a_case_with_no_question_is_an_error_row_and_never_a_call() -> None:
    task, log = _task_and_log()

    outcome = task(Case.model_validate({"id": "rq-001", "input": {"q": "wrong key"}}))

    assert outcome.error is not None and "no question" in outcome.error
    assert log.counted == 0, "a row the task cannot attempt must not be priced or sent"


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        pytest.param(1, ["a"], id="one takes the first"),
        pytest.param(3, ["a", "c", "e"], id="three spread across the set"),
        pytest.param(5, ["a", "b", "c", "d", "e"], id="all of them"),
        pytest.param(99, ["a", "b", "c", "d", "e"], id="more than there are"),
        pytest.param(0, ["a", "b", "c", "d", "e"], id="zero is not a sample"),
    ],
)
def test_a_smoke_sample_is_spread_across_the_set_not_taken_off_the_front(
    n: int, expected: list[str]
) -> None:
    """The front of this dataset is all ITC. A sample off the front smoke-tests one report."""
    cases = [Case.model_validate({"id": i, "input": {"question": i}}) for i in "abcde"]

    assert [case.id for case in spread(cases, n)] == expected
