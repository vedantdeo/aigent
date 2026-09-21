"""Parallelization: sections that land under the right report, and votes under the right reviewer,
whichever order the threads finish in."""

from __future__ import annotations

import re

import pytest

from entropic.workflows.parallelization import REVIEWERS, Vote, run
from entropic.workflows.reports import REPORTS

from ..conftest import MakeLlm, Sent
from .conftest import FakeSearch


def _reviewer(sent: Sent) -> str:
    return next(name for name, check in REVIEWERS.items() if check in str(sent.system))


def test_each_report_gets_its_own_call_on_its_own_passages(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    def reply(sent: Sent) -> Vote | str:
        if sent.schema is Vote:
            return Vote(passed=True, reason="fine")
        return sent.prompt.splitlines()[0]  # "company: <name>"

    llm, fake = make_llm(reply)

    result = run(llm, search, "water?")

    assert result.sections == {doc_id: f"company: {name}" for doc_id, name in REPORTS.items()}
    assert search.asked == [("water?", doc_id) for doc_id in REPORTS]
    for sent in fake.sent:
        if sent.kind == "create":
            cited = {found.split("#")[0] for found in re.findall(r'id="([^"]+)"', sent.prompt)}
            assert len(cited) == 1, f"a section saw passages from {cited}"


@pytest.mark.parametrize(
    ("failing", "flagged"),
    [
        pytest.param(set[str](), False, id="every reviewer passes"),
        pytest.param({"figures"}, True, id="one veto flags the answer"),
        pytest.param(set(REVIEWERS), True, id="every reviewer fails"),
    ],
)
def test_each_reviewer_is_a_veto(
    make_llm: MakeLlm, search: FakeSearch, failing: set[str], flagged: bool
) -> None:
    def reply(sent: Sent) -> Vote | str:
        if sent.schema is not Vote:
            return "a section"
        name = _reviewer(sent)
        return Vote(passed=name not in failing, reason=name)

    llm, _ = make_llm(reply)

    result = run(llm, search, "water?")

    assert result.flagged is flagged
    assert {name: vote.reason for name, vote in result.votes.items()} == {n: n for n in REVIEWERS}
