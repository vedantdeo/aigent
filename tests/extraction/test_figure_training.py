"""The figure fine-tune's training data: which chunks are asked about, the question and answer
requests, the dry runs that spend nothing, and the conversations `build` writes from the answers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from aigent.config import FIGURE_ANSWER_CACHE, FIGURE_LABEL_MODEL
from aigent.extraction import figure_training
from aigent.extraction.figure_training import (
    QUESTION_SYSTEM,
    Answer,
    Questions,
    answer_request,
    ask_request,
    build,
    select,
    source_of,
)
from aigent.extraction.figures import METRICS, SYSTEM, TOOL, Labelled
from aigent.llm.core import Request
from aigent.llm.messages import Failed, Parsed, Usage
from aigent.retrieval.chunk import Chunk

CORE = "Revenue from operations was 1,234 for the year."
SCALED = "The plant added capacity worth Rs 1,250 crore in the year."
PLAIN = "The bank opened 1,234 branches across the country."
YEARS_ONLY = "The policy was revised in 2019 and again in 2023."


def chunk(doc_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(id=f"{doc_id}#{ordinal:04d}", text=text, doc_id=doc_id, ordinal=ordinal, start=0)


@pytest.fixture
def chunks() -> list[Chunk]:
    rows: list[Chunk] = []
    for doc_id in ("AAA-FY25", "BBB-FY25"):
        texts = [CORE] * 3 + [SCALED] * 3 + [PLAIN] * 3 + [YEARS_ONLY] * 2
        rows += [chunk(doc_id, i, t) for i, t in enumerate(texts)]
    return rows


def test_select_samples_each_kind_per_report_and_skips_chunks_with_only_years(
    chunks: list[Chunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, n in (("CORE", 2), ("SCALED", 1), ("PLAIN", 1)):
        monkeypatch.setattr(figure_training, f"FIGURE_LABEL_{name}_PER_DOC", n)

    picked = select(chunks, seed=7)

    for doc_id in ("AAA-FY25", "BBB-FY25"):
        texts = [c.text for c in picked if c.doc_id == doc_id]
        counts = tuple(texts.count(t) for t in (CORE, SCALED, PLAIN, YEARS_ONLY))
        assert counts == (2, 1, 1, 0), doc_id
    assert [c.id for c in picked] == sorted(c.id for c in picked)
    assert [c.id for c in picked] == [c.id for c in select(chunks, seed=7)], "seeded"


def test_select_takes_what_there_is_when_a_report_has_too_few(chunks: list[Chunk]) -> None:
    assert len(select(chunks)) == 18, "all three of each kind in both reports"


def test_an_ask_request_carries_the_question_rules_and_the_passage() -> None:
    request = ask_request(chunk("AAA-FY25", 3, CORE), "Aaa Limited, Annual Report", 256)

    assert (request.model, request.system, request.max_tokens) == (
        FIGURE_LABEL_MODEL,
        QUESTION_SYSTEM,
        256,
    )
    assert request.step == "ask:AAA-FY25#0003"
    user = str(request.messages[0]["content"])
    assert CORE in user and "Aaa Limited, Annual Report" in user
    for metric in METRICS:
        assert (metric in QUESTION_SYSTEM) is (metric != "other"), metric


def test_an_answer_request_is_the_fine_tunes_own_prompt_marked_for_the_cache() -> None:
    request = answer_request(CORE, "Aaa Limited", "What was revenue?", "answer:x", 512)

    assert request.model == FIGURE_LABEL_MODEL
    assert request.system == [
        {"type": "text", "text": SYSTEM, "cache_control": FIGURE_ANSWER_CACHE}
    ]
    user = str(request.messages[0]["content"])
    assert "<wanted>What was revenue?</wanted>" in user and CORE in user


def test_the_teachers_schema_holds_every_labelled_field_and_only_the_calls_rule() -> None:
    taught = figure_training.Taught.model_fields
    assert set(taught) == set(Labelled.model_fields)
    assert [n for n, f in taught.items() if f.description] == ["call"]


def test_a_report_names_itself_by_company_and_title() -> None:
    assert source_of("WIPRO-FY25") == "Wipro Limited, Integrated Annual Report 2024-25"


class FakeLlm:
    """Counts for free and refuses to send unless given replies; records each batch sent."""

    client = "anthropic"
    spent_usd = 0.01

    def __init__(
        self, replies: Sequence[BaseModel | None] | None = None, cache_reads: int = 0
    ) -> None:
        self.replies = list(replies) if replies is not None else None
        self.batches: list[int] = []
        self.parsed: list[Request] = []
        self.cache_reads = cache_reads

    def parse(self, request: Request, schema: type[BaseModel]) -> Parsed[BaseModel]:
        if self.replies is None:
            raise AssertionError("a dry run sent a call")
        self.parsed.append(request)
        reply, self.replies = self.replies[0], self.replies[1:]
        assert reply is not None
        usage = Usage(input_tokens=1, output_tokens=1, cache_write_tokens=100)
        return Parsed(text="", usage=usage, stop_reason="end_turn", model="m", parsed=reply)

    def count(self, request: Request, schema: type[BaseModel] | None = None) -> int:
        return 100 if request.system is not None else 40

    def batch_records(
        self, requests: Sequence[Request], schema: type[BaseModel]
    ) -> list[Parsed[BaseModel] | Failed]:
        if self.replies is None:
            raise AssertionError("a dry run sent the batch")
        self.batches.append(len(requests))
        usage = Usage(input_tokens=1, output_tokens=1, cache_read_tokens=self.cache_reads)
        taken, self.replies = self.replies[: len(requests)], self.replies[len(requests) :]
        return [
            Failed("errored")
            if reply is None
            else Parsed(text="", usage=usage, stop_reason="end_turn", model="m", parsed=reply)
            for reply in taken
        ]


def _args(yes: bool) -> argparse.Namespace:
    return argparse.Namespace(yes=yes, sample=None, client="anthropic")


@pytest.fixture
def files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    picked = [chunk("WIPRO-FY25", 1, SCALED), chunk("WIPRO-FY25", 2, PLAIN)]
    monkeypatch.setattr(figure_training, "_chunks", lambda: picked)
    monkeypatch.setattr(figure_training, "select", lambda chunks: list(chunks))
    monkeypatch.setattr(figure_training, "QUESTIONS", tmp_path / "questions.jsonl")
    monkeypatch.setattr(figure_training, "ANSWERS", tmp_path / "answers.jsonl")
    return tmp_path


@pytest.mark.parametrize("step", ["ask", "answer"])
def test_a_dry_run_prints_the_worst_case_and_sends_nothing(
    step: str, files: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (files / "questions.jsonl").write_text(
        json.dumps({"chunk_id": "c", "source": "s", "passage": "p", "questions": ["q"]}) + "\n"
    )
    monkeypatch.setattr(figure_training, "_llm", lambda client: FakeLlm())

    getattr(figure_training, step)(_args(yes=False))

    out = capsys.readouterr().out
    assert "worst case $" in out and "nothing spent" in out
    assert not (files / "answers.jsonl").exists()


def test_asking_writes_each_chunks_questions_and_skips_the_ones_without_a_record(
    files: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeLlm([Questions(questions=["What capacity did Wipro add?"]), None])
    monkeypatch.setattr(figure_training, "_llm", lambda client: fake)

    figure_training.ask(_args(yes=True))

    rows = [json.loads(x) for x in (files / "questions.jsonl").read_text().splitlines()]
    assert [r["chunk_id"] for r in rows] == ["WIPRO-FY25#0001"]
    assert rows[0]["questions"] == ["What capacity did Wipro add?"]
    assert rows[0]["source"].startswith("Wipro Limited") and rows[0]["passage"] == SCALED
    assert "spent $0.0100" in capsys.readouterr().out


def _taught(value: float, call: str | None) -> dict[str, object]:
    return {
        "metric": "capex",
        "value": value,
        "unit": "INR",
        "year": 2025,
        "description": None,
        "call": call,
    }


@pytest.mark.parametrize(
    ("cache_reads", "said"),
    [
        pytest.param(808, "cache read by 2 of 2", id="the batch read the warmed cache"),
        pytest.param(0, "cache read by 0 of 2", id="the batch missed it"),
    ],
)
def test_answering_warms_the_cache_with_the_first_question_then_batches_the_rest(
    cache_reads: int,
    said: str,
    files: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(figure_training, "FIGURE_ANSWER_SLICE", 2)
    row = {"chunk_id": "WIPRO-FY25#0001", "source": "Wipro Limited", "passage": SCALED}
    (files / "questions.jsonl").write_text(
        json.dumps(row | {"questions": ["q1", "q2", "q3", "q4"]}) + "\n"
    )
    good = Answer.model_validate({"figures": [_taught(12500000000, "1250 * 10**7")]})
    fake = FakeLlm([good, good, None, good], cache_reads=cache_reads)
    monkeypatch.setattr(figure_training, "_llm", lambda client: fake)

    figure_training.answer(_args(yes=True))

    assert [str(r.messages[0]["content"]).endswith("<wanted>q1</wanted>") for r in fake.parsed] == [
        True
    ], "one ordinary call, for the first question"
    assert fake.batches == [2, 1], "the other three in slices of at most two"
    rows = [json.loads(x) for x in (files / "answers.jsonl").read_text().splitlines()]
    assert [r["question"] for r in rows] == ["q1", "q2", "q4"]
    assert said in capsys.readouterr().out
    assert rows[0]["figures"][0]["call"] == "1250 * 10**7" and rows[0]["passage"] == SCALED


def _answer(chunk_id: str, question: str, *figures: dict[str, object]) -> str:
    passage = f"Capex of Rs 1,250 crore, in {chunk_id}"
    row = {"chunk_id": chunk_id, "source": "Aaa Limited", "passage": passage}
    return json.dumps(row | {"question": question, "figures": list(figures)})


GOOD = _taught(12500000000, "1250 * 10**7")
WRONG = _taught(1250, "1250 * 10**7")
BROKEN: dict[str, object] = {"metric": "capex", "call": "1250 * 10**7"}


def test_build_writes_one_conversation_per_good_answer_and_drops_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figure_training, "FIGURE_VALID_FRACTION", 0.5)
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "\n".join(
            [
                _answer("AAA#0001", "good", GOOD),
                _answer("AAA#0001", "a call that does not give the value", WRONG),
                _answer("AAA#0001", "no figures"),
                _answer("AAA#0001", "a figure missing fields", BROKEN),
                _answer("AAA#0002", "good too", GOOD),
            ]
        )
        + "\n"
    )

    built = build(answers, tmp_path / "out", seed=0)

    assert (built.train, built.valid, built.dropped) == (1, 1, 3)
    for name in ("train", "valid"):
        lines = (tmp_path / "out" / f"{name}.jsonl").read_text().splitlines()
        rows = [json.loads(x) for x in lines]
        assert len(rows) == 1 and rows[0]["tools"] == [TOOL]
        assert rows[0]["messages"][0]["role"] == "system"


def test_build_keeps_every_question_of_a_chunk_on_one_side(tmp_path: Path) -> None:
    answers = tmp_path / "answers.jsonl"
    rows = [_answer(f"AAA#{i:04d}", q, GOOD) for i in range(20) for q in ("q1", "q2")]
    answers.write_text("\n".join(rows))

    build(answers, tmp_path / "out", seed=3)

    sides: dict[str, set[str]] = {}
    for name in ("train", "valid"):
        for line in (tmp_path / "out" / f"{name}.jsonl").read_text().splitlines():
            passage = json.loads(line)["messages"][1]["content"].split("<passage>")[1]
            sides.setdefault(passage, set()).add(name)
    assert all(len(s) == 1 for s in sides.values())
