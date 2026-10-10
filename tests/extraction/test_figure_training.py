"""The figure fine-tune's training data: which chunks are labelled, the labeller's request, the dry
run that spends nothing, and the conversations `build` writes from the labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from aigent.config import FIGURE_LABEL_MODEL
from aigent.extraction import figure_training
from aigent.extraction.figure_training import (
    LABEL_SYSTEM,
    Labels,
    build,
    label_request,
    select,
    source_of,
)
from aigent.extraction.figures import METRICS, TOOL, Labelled
from aigent.llm.core import Request
from aigent.llm.messages import Failed, Parsed, Usage
from aigent.retrieval.chunk import Chunk

SCALED = "Revenue grew to Rs 1,250 crore in the year."
PLAIN = "The bank opened 1,234 branches across the country."
YEARS_ONLY = "The policy was revised in 2019 and again in 2023."


def chunk(doc_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(id=f"{doc_id}#{ordinal:04d}", text=text, doc_id=doc_id, ordinal=ordinal, start=0)


@pytest.fixture
def chunks() -> list[Chunk]:
    rows: list[Chunk] = []
    for doc_id in ("AAA-FY25", "BBB-FY25"):
        texts = [SCALED] * 4 + [PLAIN] * 4 + [YEARS_ONLY] * 3
        rows += [chunk(doc_id, i, t) for i, t in enumerate(texts)]
    return rows


def test_select_samples_each_kind_per_report_and_skips_chunks_with_only_years(
    chunks: list[Chunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figure_training, "FIGURE_LABEL_SCALED_PER_DOC", 2)
    monkeypatch.setattr(figure_training, "FIGURE_LABEL_PLAIN_PER_DOC", 1)

    picked = select(chunks, seed=7)

    for doc_id in ("AAA-FY25", "BBB-FY25"):
        texts = [c.text for c in picked if c.doc_id == doc_id]
        assert (texts.count(SCALED), texts.count(PLAIN), texts.count(YEARS_ONLY)) == (2, 1, 0)
    assert [c.id for c in picked] == sorted(c.id for c in picked)
    assert [c.id for c in picked] == [c.id for c in select(chunks, seed=7)], "seeded"


def test_select_takes_what_there_is_when_a_report_has_too_few(chunks: list[Chunk]) -> None:
    assert len(select(chunks)) == 16, "all four of each kind in both reports"


def test_a_label_request_carries_the_rules_the_passage_and_the_labelling_model() -> None:
    request = label_request(chunk("AAA-FY25", 3, SCALED), "Aaa Limited, Annual Report", 512)

    assert request.model == FIGURE_LABEL_MODEL
    assert request.system == LABEL_SYSTEM and request.max_tokens == 512
    assert request.step == "label:AAA-FY25#0003"
    user = str(request.messages[0]["content"])
    assert SCALED in user and "Aaa Limited, Annual Report" in user
    for name, field in Labelled.model_fields.items():
        assert field.description and field.description in LABEL_SYSTEM, f"{name} rule missing"
    for metric in METRICS:
        assert (metric in LABEL_SYSTEM) is True, metric


def test_a_report_names_itself_by_company_and_title() -> None:
    assert source_of("WIPRO-FY25") == "Wipro Limited, Integrated Annual Report 2024-25"


class FakeLlm:
    """Counts for free and refuses to send unless given replies."""

    client = "anthropic"

    def __init__(self, replies: Sequence[Labels | None] | None = None) -> None:
        self.replies = replies
        self.sent: list[Request] = []

    def count(self, request: Request, schema: type[BaseModel] | None = None) -> int:
        return 100

    def batch_records(
        self, requests: Sequence[Request], schema: type[Labels]
    ) -> list[Parsed[Labels] | Failed]:
        if self.replies is None:
            raise AssertionError("a dry run sent the batch")
        self.sent = list(requests)
        usage = Usage(input_tokens=1, output_tokens=1)
        return [
            Failed("errored")
            if reply is None
            else Parsed(text="", usage=usage, stop_reason="end_turn", model="m", parsed=reply)
            for reply in self.replies
        ]


def _args(yes: bool) -> argparse.Namespace:
    return argparse.Namespace(yes=yes, sample=None, client="anthropic")


def _labels(question: str, amount: str, value: float) -> Labels:
    figure = {"metric": "revenue", "value": value, "unit": "INR", "year": 2025, "amount": amount}
    return Labels.model_validate({"asked": [{"question": question, "figures": [figure]}]})


@pytest.fixture
def labelling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[pytest.MonkeyPatch, Path]:
    picked = [chunk("WIPRO-FY25", 1, SCALED), chunk("WIPRO-FY25", 2, PLAIN)]
    monkeypatch.setattr(figure_training, "_chunks", lambda: picked)
    monkeypatch.setattr(figure_training, "select", lambda chunks: list(chunks))
    monkeypatch.setattr(figure_training, "LABELS", tmp_path / "labels.jsonl")
    return monkeypatch, tmp_path


def test_a_dry_run_prints_the_worst_case_and_sends_nothing(
    labelling: tuple[pytest.MonkeyPatch, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch, tmp_path = labelling
    monkeypatch.setattr(figure_training, "_llm", lambda client: FakeLlm())

    figure_training.label(_args(yes=False))

    out = capsys.readouterr().out
    assert "2 chunks from 1 reports" in out and "worst case $" in out and "nothing spent" in out
    assert not (tmp_path / "labels.jsonl").exists()


def test_sending_writes_each_labelled_chunk_and_counts_the_ones_without_a_record(
    labelling: tuple[pytest.MonkeyPatch, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch, tmp_path = labelling
    fake = FakeLlm([_labels("What was revenue?", "Rs 1,250 crore", 12500000000), None])
    monkeypatch.setattr(figure_training, "_llm", lambda client: fake)

    figure_training.label(_args(yes=True))

    rows = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert [r["chunk_id"] for r in rows] == ["WIPRO-FY25#0001"]
    assert rows[0]["passage"] == SCALED and rows[0]["source"].startswith("Wipro Limited")
    assert rows[0]["asked"][0]["figures"][0]["amount"] == "Rs 1,250 crore"
    assert "1 chunks, 1 without a record" in capsys.readouterr().out
    assert len(fake.sent) == 2


def _row(chunk_id: str, *asked: dict[str, object]) -> str:
    passage = f"passage of {chunk_id}"
    return json.dumps(
        {"chunk_id": chunk_id, "source": "Aaa Limited", "passage": passage, "asked": list(asked)}
    )


def _asked(question: str, *figures: dict[str, object]) -> dict[str, object]:
    return {"question": question, "figures": list(figures)}


GOOD: dict[str, object] = {
    "metric": "revenue",
    "value": 12500000000,
    "unit": "INR",
    "year": 2025,
    "amount": "1,250 crore",
}
WRONG: dict[str, object] = GOOD | {"value": 1250}
BROKEN: dict[str, object] = {"metric": "revenue", "amount": "1,250 crore"}


def test_build_writes_one_conversation_per_good_question_and_drops_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figure_training, "FIGURE_VALID_FRACTION", 0.5)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            [
                _row(
                    "AAA#0001",
                    _asked("good", GOOD),
                    _asked("arithmetic that does not check out", WRONG),
                    _asked("no figures"),
                    _asked("a figure missing fields", BROKEN),
                ),
                _row("AAA#0002", _asked("good too", GOOD)),
            ]
        )
        + "\n"
    )

    built = build(labels, tmp_path / "out", seed=0)

    assert (built.train, built.valid, built.dropped) == (1, 1, 3)
    for name in ("train", "valid"):
        rows = [
            json.loads(x) for x in (tmp_path / "out" / f"{name}.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1 and rows[0]["tools"] == [TOOL]
        assert rows[0]["messages"][0]["role"] == "system"


def test_build_keeps_every_question_of_a_chunk_on_one_side(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(_row(f"AAA#{i:04d}", _asked("q1", GOOD), _asked("q2", GOOD)) for i in range(20))
    )

    build(labels, tmp_path / "out", seed=3)

    sides = {}
    for name in ("train", "valid"):
        for line in (tmp_path / "out" / f"{name}.jsonl").read_text().splitlines():
            user = json.loads(line)["messages"][1]["content"]
            sides.setdefault(user, set()).add(name)
    assert all(len(s) == 1 for s in sides.values())
