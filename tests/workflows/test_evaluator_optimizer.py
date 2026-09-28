"""Evaluator-optimizer: the loop ends at a pass or at the cap, and a revision sees why it failed."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from aigent.config import JUDGE_MODEL, MAX_REFINE_ROUNDS, MODEL
from aigent.workflows.evaluator_optimizer import Critique, run

from ..conftest import FakeSearch, MakeLlm, Reply, Sent

PROBLEM = "claim 2 cites no passage"


def scripted(verdicts: Sequence[bool]) -> Reply:
    """Drafts numbered in order, and one verdict per evaluation from `verdicts`."""
    remaining = iter(verdicts)
    drafted = 0

    def reply(sent: Sent) -> Critique | str:
        nonlocal drafted
        if sent.schema is Critique:
            passed = next(remaining)
            return Critique(passed=passed, problems=[] if passed else [PROBLEM])
        drafted += 1
        return f"draft {drafted}"

    return reply


@pytest.mark.parametrize(
    ("verdicts", "passed"),
    [
        pytest.param([True], True, id="a first draft that passes ends the loop"),
        pytest.param([False, True], True, id="one revision"),
        pytest.param(
            [False] * MAX_REFINE_ROUNDS, False, id="the cap ends a loop that never passes"
        ),
    ],
)
def test_the_loop_runs_until_a_pass_or_the_cap(
    make_llm: MakeLlm, search: FakeSearch, verdicts: list[bool], passed: bool
) -> None:
    llm, fake = make_llm(scripted(verdicts))

    result = run(llm, search, "the question")

    assert result.passed is passed
    assert result.answer == f"draft {len(verdicts)}"
    assert [critique.passed for critique in result.critiques] == verdicts
    assert [sent.model for sent in fake.sent] == [MODEL, JUDGE_MODEL] * len(verdicts), (
        "the evaluator is never the model that drafted"
    )


def test_a_revision_sees_its_last_draft_and_what_failed(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    llm, fake = make_llm(scripted([False, True]))

    run(llm, search, "the question")

    first, second = (sent.prompt for sent in fake.sent if sent.model == MODEL)
    assert PROBLEM not in first
    assert "draft 1" in second and PROBLEM in second
