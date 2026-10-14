"""The analyst's run: who the generalist is, what a run can cost, that nothing is sent without
--yes, and that a question's transcript holds every step and the result."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from aigent import tracing  # noqa: E402
from aigent.analyst import run as run_module  # noqa: E402
from aigent.analyst.memo import Claim, Judged, Memo  # noqa: E402
from aigent.analyst.run import (  # noqa: E402
    Generalist,
    analyse,
    generalist_of,
    main,
    worst_case_usd,
)
from aigent.config import (  # noqa: E402
    GENERALIST_MODEL,
    MAX_USD_ANALYST_CRITIC,
    MAX_USD_ANALYST_GENERALIST,
    MAX_USD_ANALYST_SPECIALIST,
)

from ..conftest import MakeLlm, Sent, tool_turn  # noqa: E402
from .conftest import DIVIDEND, box, search  # noqa: E402

DRAFT = f"Reliance recommended Rs 5.50 a share [{DIVIDEND}]."
MEMO = Memo(
    summary="Reliance recommended Rs 5.50.",
    claims=[Claim(id="c1", text="Reliance recommended Rs 5.50 a share.", citations=[DIVIDEND])],
    caveats=["The record date is not given."],
)


@pytest.mark.parametrize(
    ("named", "thinking", "expected"),
    [
        pytest.param(
            f"anthropic:{GENERALIST_MODEL}",
            "adaptive",
            Generalist("anthropic", GENERALIST_MODEL, {"type": "adaptive"}),
            id="the default: Sonnet thinking adaptively",
        ),
        pytest.param(
            "anthropic:claude-haiku-4-5",
            "2048",
            Generalist("anthropic", "claude-haiku-4-5", {"type": "enabled", "budget_tokens": 2048}),
            id="a model with a fixed thinking budget",
        ),
        pytest.param(
            "sarvam-thinking", "off", Generalist("sarvam-thinking", None, None), id="a client's own"
        ),
    ],
)
def test_a_generalist_is_a_client_a_model_and_how_it_thinks(
    named: str, thinking: str, expected: Generalist
) -> None:
    assert generalist_of(named, thinking) == expected


@pytest.mark.parametrize(
    ("named", "thinking"),
    [("nobody", "off"), ("sarvam", "lots")],
    ids=["an unknown client", "a budget that is not a number"],
)
def test_a_generalist_that_cannot_be_built_is_refused(named: str, thinking: str) -> None:
    with pytest.raises(ValueError):
        generalist_of(named, thinking)


def test_the_worst_case_is_every_ceiling_a_run_holds() -> None:
    each = MAX_USD_ANALYST_GENERALIST + MAX_USD_ANALYST_CRITIC
    assert worst_case_usd(3) == pytest.approx(3 * each + 2 * MAX_USD_ANALYST_SPECIALIST)


def _never(*args: object, **kwargs: object) -> None:
    raise AssertionError("something was sent")


def test_without_yes_a_run_estimates_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(run_module, "warm_up", _never)
    monkeypatch.setattr(run_module, "connect", _never)

    main(["--generalist", "sarvam-thinking", "--thinking", "off", "--out", str(tmp_path)])

    printed = capsys.readouterr().out
    assert f"worst case ${worst_case_usd(3):.2f}" in printed and "nothing sent" in printed
    assert list(tmp_path.iterdir()) == []


def test_a_model_with_no_price_is_refused_before_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "warm_up", _never)

    with pytest.raises(SystemExit, match="no price"):
        main(["--generalist", "anthropic:claude-unknown", "--yes"])


def test_a_question_s_transcript_holds_each_step_and_the_result(
    make_llm: MakeLlm, tmp_path: Path
) -> None:
    def reply(sent: Sent) -> object:
        if sent.schema is Memo:
            return MEMO
        return (
            tool_turn(("s1", "search_filings", {"query": "dividend"}))
            if len(sent.messages) == 1
            else DRAFT
        )

    generalist, _ = make_llm(reply)
    critic, _ = make_llm(lambda sent: Judged(verdict="supported", reason="it says Rs 5.50"))
    path = tmp_path / "q1.md"
    chosen = Generalist("anthropic", GENERALIST_MODEL, None)

    done = analyse(
        "What dividend did Reliance recommend?", chosen, generalist, critic, box(search), path
    )

    text = path.read_text(encoding="utf-8")
    steps = [
        "# What dividend did Reliance recommend?",
        "## generation: research",
        "**Asked for** `search_filings`",
        "## retriever: search_filings",
        f"**Said:**\n\n{DRAFT}",
        "## generation: memo",
        "## generation: critic",
        "## Result",
        "**Outcome:** 1/1 claims supported, 0 redrafts",
        f"- c1 Reliance recommended Rs 5.50 a share. `{DIVIDEND}` — **supported**",
        "**Caveats:** The record date is not given.",
    ]
    at = [text.find(step) for step in steps]
    assert -1 not in at and at == sorted(at), dict(zip(steps, at, strict=True))
    assert done.memo == MEMO
    assert not isinstance(tracing.tracer(), run_module.FileTracer), "the tracer is put back"
