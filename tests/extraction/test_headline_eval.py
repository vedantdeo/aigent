"""Specification for extraction/headline_eval.py: the labelled headlines exported as the training
prompt, and saved replies graded as live ones are."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from aigent.config import SYNTHETIC_CLIENT, max_tokens
from aigent.evals.dataset import Case
from aigent.evals.runner import EvalRun
from aigent.extraction import headline_eval
from aigent.extraction.headline_eval import export, replay_task
from aigent.extraction.headlines import Extraction, Resolver
from aigent.extraction.synthetic import Synthetic, render, wire_for
from aigent.llm.adapters import spec

HL1 = Case(
    id="hl-001",
    input={"headline": "Infosys Q3 net profit rises 11.5% to Rs 6,806 crore"},
    expected={
        "ticker": "INFY",
        "metric": "profit",
        "quarter": "Q3",
        "direction": "up",
        "change_pct": 11.5,
    },
)
UNLABELLED = Case(id="hl-099", input={"headline": "Sensex ends flat"})
GOOD = json.dumps(
    {
        "company": "Infosys",
        "metric": "profit",
        "quarter": "Q3",
        "direction": "up",
        "change_pct": 11.5,
    }
)


def test_export_writes_each_labelled_case_as_its_training_prompt() -> None:
    (row,) = export([HL1, UNLABELLED])

    label = Extraction.model_validate_json(GOOD)
    example = Synthetic(headline=str(HL1.input["headline"]), label=label, rule="plain")
    trained = render(
        example, wire_for(SYNTHETIC_CLIENT), max_tokens("HEADLINE", spec(SYNTHETIC_CLIENT))
    )
    assert row == {"id": "hl-001", "messages": trained["messages"][:-1]}


@pytest.mark.parametrize(
    ("prediction", "error"),
    [
        pytest.param(None, "no prediction for hl-001", id="a case with no prediction"),
        pytest.param(
            {"id": "hl-001", "final": None, "error": "CUDA out of memory"},
            "CUDA out of memory",
            id="a failed prediction keeps its error",
        ),
        pytest.param(
            {"id": "hl-001", "final": None, "error": None},
            "no final reply",
            id="no reply and no error",
        ),
        pytest.param(
            {"id": "hl-001", "final": "Infosys, profit, up", "error": None},
            "no parsed output",
            id="a reply that is not the record",
        ),
    ],
)
def test_a_case_without_a_readable_reply_is_an_error(
    prediction: dict[str, JsonValue] | None, error: str
) -> None:
    saved = {} if prediction is None else {"hl-001": prediction}

    outcome = replay_task(saved, "qwen", Resolver())(HL1)

    assert outcome.error is not None and error in outcome.error, outcome.error
    assert outcome.output == {} and outcome.usage is None


@pytest.mark.parametrize(
    ("company", "ticker"),
    [
        pytest.param("Infosys", "INFY", id="a company resolved to its ticker"),
        pytest.param(None, None, id="no company, no ticker"),
    ],
)
def test_a_reply_is_read_and_resolved_as_a_live_one(
    company: str | None, ticker: str | None
) -> None:
    final = json.dumps({**json.loads(GOOD), "company": company})

    outcome = replay_task({"hl-001": {"final": final}}, "qwen", Resolver())(HL1)

    assert outcome.error is None and outcome.model == "qwen"
    assert outcome.output["ticker"] == ticker and outcome.output["metric"] == "profit"
    assert outcome.raw == final


def test_grade_runs_only_the_labelled_cases_through_the_graders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": "hl-001", "final": GOOD, "error": None}) + "\n")
    runs: list[EvalRun] = []
    monkeypatch.setattr(headline_eval, "load_jsonl", lambda _: [HL1, UNLABELLED])
    monkeypatch.setattr(headline_eval, "write_report", lambda run: runs.append(run) or tmp_path)

    headline_eval.main(["grade", str(predictions), "--name", "qwen"])

    (run,) = runs
    (row,) = run.rows
    assert run.model == "qwen" and run.spent_usd == 0.0
    assert row.case_id == "hl-001" and row.passed, row.scores


def test_export_writes_the_fifty_prompts(tmp_path: Path) -> None:
    out = tmp_path / "test.jsonl"

    headline_eval.main(["export", "--out", str(out)])

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 50 and [m["role"] for m in rows[0]["messages"]] == ["system", "user"]
