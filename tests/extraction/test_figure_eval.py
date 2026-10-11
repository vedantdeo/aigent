"""The figure eval: the test set exported as the fine-tune's own prompts, and predictions graded for
their figures and for when they called calculate."""

from __future__ import annotations

import json

import pytest
from pydantic import JsonValue

from aigent.evals.dataset import Case
from aigent.extraction.figure_eval import export, grade, needs_call
from aigent.extraction.figures import SYSTEM, TOOL

PASSAGE = "Revenue was Rs 1,250 crore; the company runs 41 plants."
SCALED: dict[str, JsonValue] = {
    "metric": "revenue",
    "value": 12500000000,
    "unit": "INR",
    "year": 2025,
    "description": None,
}
PLAIN: dict[str, JsonValue] = {
    "metric": "other",
    "value": 41,
    "unit": None,
    "year": None,
    "description": "plants",
}


def case(case_id: str, figure: dict[str, JsonValue]) -> Case:
    return Case(
        id=case_id,
        input={"chunk_id": "ITC-FY25#0010", "wanted": f"question {case_id}"},
        expected={"figures": [figure]},
    )


def test_export_writes_each_case_as_the_fine_tunes_prompt_with_its_passage_and_label() -> None:
    (row,) = export([case("fg-1", SCALED)], {"ITC-FY25#0010": PASSAGE})

    messages = json.loads(json.dumps(row["messages"]))
    assert messages[0] == {"role": "system", "content": SYSTEM}
    user = messages[1]["content"]
    assert PASSAGE in user and "<wanted>question fg-1</wanted>" in user
    assert "ITC Limited" in user, "the source names the company"
    assert row["tools"] == [TOOL] and row["passage"] == PASSAGE and row["expected"] == [SCALED]


def _row(case_id: str, figure: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return export([case(case_id, figure)], {"ITC-FY25#0010": PASSAGE})[0]


@pytest.mark.parametrize(
    ("figure", "needed"),
    [
        pytest.param(SCALED, True, id="a crore amount needs a call"),
        pytest.param(PLAIN, False, id="a number printed in full does not"),
    ],
)
def test_needs_call(figure: dict[str, JsonValue], needed: bool) -> None:
    assert needs_call(_row("fg-x", figure)) is needed


def answer(*figures: dict[str, JsonValue]) -> str:
    return json.dumps({"figures": list(figures)})


def test_grade_counts_figures_json_and_tool_use_per_arm() -> None:
    rows = [_row("fg-1", SCALED), _row("fg-2", PLAIN), _row("fg-3", SCALED), _row("fg-4", PLAIN)]
    predictions: dict[str, dict[str, JsonValue]] = {
        "fg-1": {"final": answer(SCALED), "calls": ["1250 * 10**7"]},
        "fg-2": {"final": answer(PLAIN), "calls": ["41 * 1"]},
        "fg-3": {"final": "not json", "calls": []},
    }

    graded = grade(rows, predictions)

    assert (graded.cases, graded.passed, graded.parsed) == (4, 2, 2)
    assert graded.recall == pytest.approx(0.5)
    assert (graded.called_when_needed, graded.needed) == (1, 2)
    assert (graded.called_when_not, graded.not_needed) == (1, 2)
    assert "passed 2/4" in graded.lines()[0]


def test_a_wrong_figure_fails_and_says_which_part() -> None:
    wrong = SCALED | {"value": 1250}
    graded = grade([_row("fg-1", SCALED)], {"fg-1": {"final": answer(wrong), "calls": []}})

    assert (graded.passed, graded.all_found, graded.no_extra, graded.parsed) == (0, 0, 0, 1)


def test_a_description_not_in_the_passage_fails_the_described_part() -> None:
    made_up = PLAIN | {"description": "factories"}
    graded = grade([_row("fg-2", PLAIN)], {"fg-2": {"final": answer(made_up), "calls": []}})

    assert (graded.passed, graded.all_found, graded.described) == (0, 1, 0)
