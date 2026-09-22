"""Orchestrator-workers: the plan decides how many workers run, and code only caps it.

What separates this from sectioning is that the split is data the model wrote, so the table's
rows are plans of different sizes rather than inputs of different kinds.
"""

from __future__ import annotations

import pytest

from entropic.config import MAX_PLAN_TASKS
from entropic.workflows.orchestrator_workers import SYNTHESISE, WORK, Plan, Subtask, run

from ..conftest import FakeSearch, MakeLlm, Sent


def plan_of(n: int) -> Plan:
    return Plan(
        subtasks=[
            Subtask(brief=f"brief {i}", query=f"query {i}", report="ITC-FY25" if i % 2 else None)
            for i in range(n)
        ]
    )


@pytest.mark.parametrize(
    ("planned", "workers"),
    [
        pytest.param(1, 1, id="a narrow question gets one worker"),
        pytest.param(3, 3, id="the plan decides how many"),
        pytest.param(MAX_PLAN_TASKS + 3, MAX_PLAN_TASKS, id="code caps a plan that runs long"),
    ],
)
def test_the_plan_decides_the_workers_and_code_caps_them(
    make_llm: MakeLlm, search: FakeSearch, planned: int, workers: int
) -> None:
    def reply(sent: Sent) -> Plan | str:
        if sent.schema is Plan:
            return plan_of(planned)
        if sent.system == SYNTHESISE:
            return "the answer"
        return "found for " + sent.prompt.rsplit("brief: ", 1)[1]

    llm, fake = make_llm(reply)

    result = run(llm, search, "the question")

    ran = plan_of(planned).subtasks[:workers]
    assert [sent.system for sent in fake.sent].count(WORK) == workers
    assert search.asked == [(task.query, task.report) for task in ran]
    assert result.findings == [f"found for {task.brief}" for task in ran]
    synthesis = fake.sent[-1]
    assert synthesis.system == SYNTHESISE and result.answer == "the answer"
    assert all(finding in synthesis.prompt for finding in result.findings)


def test_an_empty_plan_costs_nothing_further(make_llm: MakeLlm, search: FakeSearch) -> None:
    llm, fake = make_llm(lambda sent: Plan(subtasks=[]))

    result = run(llm, search, "the question")

    assert len(fake.sent) == 1 and search.asked == []
    assert result.findings == []
