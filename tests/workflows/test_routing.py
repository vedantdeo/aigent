"""Routing: one table, a row per route, because every route is the same call with a different
destination."""

from __future__ import annotations

import pytest

from entropic.config import MODEL, SMALL_MODEL
from entropic.workflows.routing import DECLINED, Route, run

from ..conftest import FakeSearch, MakeLlm, Sent


@pytest.mark.parametrize(
    ("route", "answered_by"),
    [
        pytest.param(
            Route(kind="lookup", report="RELIANCE-FY25", reason="one figure"),
            SMALL_MODEL,
            id="a lookup goes to the small model, scoped to its report",
        ),
        pytest.param(
            Route(kind="analysis", report=None, reason="spans reports"),
            MODEL,
            id="an analysis goes to the large model, over every report",
        ),
        pytest.param(
            Route(kind="out_of_scope", report=None, reason="not in the reports"),
            None,
            id="out of scope is declined in code, with no second call",
        ),
    ],
)
def test_each_route_is_answered_by_its_own_model(
    make_llm: MakeLlm, search: FakeSearch, route: Route, answered_by: str | None
) -> None:
    def reply(sent: Sent) -> Route | str:
        return route if sent.schema is Route else "the answer"

    llm, fake = make_llm(reply)

    result = run(llm, search, "the question")

    assert result.model == answered_by
    assert [sent.model for sent in fake.sent] == [SMALL_MODEL] + (
        [answered_by] if answered_by else []
    )
    assert search.asked == ([("the question", route.report)] if answered_by else [])
    assert result.answer == ("the answer" if answered_by else DECLINED)
