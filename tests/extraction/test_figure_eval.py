"""The figure eval: the test set exported as the fine-tune's own prompts, and predictions graded for
their figures and for when they called calculate."""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import JsonValue

from aigent.evals.dataset import Case
from aigent.extraction.figure_eval import (
    census,
    export,
    faults,
    from_conversations,
    grade,
    needs_call,
)
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


SHARE: dict[str, JsonValue] = {
    "metric": "other",
    "value": 0.12,
    "unit": None,
    "year": None,
    "description": "share",
}


def _share_row() -> dict[str, JsonValue]:
    row = _row("fg-p", SHARE)
    return row | {"passage": "The share rose to 12% in the year."}


def said(*figures: dict[str, JsonValue], calls: list[str] | None = None) -> dict[str, JsonValue]:
    return {"final": answer(*figures), "calls": cast(JsonValue, calls or [])}


@pytest.mark.parametrize(
    ("figure", "prediction", "expected"),
    [
        pytest.param(SCALED, said(SCALED, calls=["1250 * 10**7"]), [], id="a perfect answer"),
        pytest.param(SCALED, said(SCALED), ["no call when needed"], id="right without the call"),
        pytest.param(
            PLAIN, said(PLAIN, calls=["41 * 1"]), ["call when not needed"], id="a needless call"
        ),
        pytest.param(
            SCALED,
            said(SCALED | {"value": 1250000000}, calls=["125 * 10**7"]),
            ["wrong scale"],
            id="a power of ten out",
        ),
        pytest.param(
            PLAIN, said(PLAIN | {"metric": "volume", "description": None}), ["metric"], id="metric"
        ),
        pytest.param(
            PLAIN, said(PLAIN | {"unit": "plants"}), ["unit not a currency"], id="a unit not money"
        ),
        pytest.param(
            SCALED,
            said(SCALED | {"unit": "USD"}, calls=["1250 * 10**7"]),
            ["unit"],
            id="the wrong currency",
        ),
        pytest.param(PLAIN, said(PLAIN | {"year": 2025}), ["year added"], id="a year added"),
        pytest.param(
            SCALED,
            said(SCALED | {"year": None}, calls=["1250 * 10**7"]),
            ["year dropped"],
            id="year dropped",
        ),
        pytest.param(
            SCALED,
            said(SCALED | {"year": 2024}, calls=["1250 * 10**7"]),
            ["year wrong"],
            id="year wrong",
        ),
        pytest.param(
            PLAIN, said(PLAIN | {"description": "factories"}), ["description"], id="description"
        ),
        pytest.param(PLAIN, said(), ["missed"], id="nothing found"),
        pytest.param(
            SCALED,
            said(SCALED | {"value": 3000000000}, calls=["300 * 10**7"]),
            ["missed", "extra"],
            id="an unrelated number is not a scale slip",
        ),
        pytest.param(PLAIN, said(PLAIN, PLAIN | {"value": 7}), ["extra"], id="one too many"),
        pytest.param(
            PLAIN,
            {"final": '{"figures": [{"value": 10**6}]}', "calls": []},
            ["expression in JSON"],
            id="expression",
        ),
        pytest.param(
            PLAIN, {"final": "not json", "calls": []}, ["invalid JSON"], id="not JSON at all"
        ),
    ],
)
def test_faults(
    figure: dict[str, JsonValue], prediction: dict[str, JsonValue], expected: list[str]
) -> None:
    assert faults(_row("fg-x", figure), prediction) == expected


def test_a_raw_percentage_is_named_as_such() -> None:
    found = faults(_share_row(), said(SHARE | {"value": 12}))

    assert found == ["no call when needed", "percentage not converted"]


def test_conversations_become_eval_rows_with_their_final_figures_as_the_label() -> None:
    user = (
        "<source>Aaa Limited</source>\n<passage>Capex was Rs 5 crore.</passage>\n<wanted>q</wanted>"
    )
    conversation: dict[str, JsonValue] = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "tool_calls": []},
            {"role": "tool", "content": "50000000.0"},
            {"role": "assistant", "content": answer(SCALED)},
        ],
        "tools": [TOOL],
    }

    (row,) = from_conversations([conversation])

    assert row["id"] == "va-0000" and row["passage"] == "Capex was Rs 5 crore."
    sent = cast(list[JsonValue], conversation["messages"])[:2]
    assert json.loads(json.dumps(row["messages"])) == sent
    assert row["expected"] == [SCALED] and row["tools"] == [TOOL]
    assert row["calls"] == [], "the label made no call"
    assert "augmented" not in row, "unmarked without the originals"


@pytest.mark.parametrize(
    ("originals", "augmented"),
    [
        pytest.param({("Capex was Rs 5 crore.", "q")}, False, id="labelled by hand"),
        pytest.param({("Capex was Rs 9 crore.", "q")}, True, id="a rewritten copy"),
    ],
)
def test_rows_say_whether_they_are_augmented(
    originals: set[tuple[str, str]], augmented: bool
) -> None:
    user = "<source>A</source>\n<passage>Capex was Rs 5 crore.</passage>\n<wanted>q</wanted>"
    conversation: dict[str, JsonValue] = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer(SCALED)},
        ],
        "tools": [TOOL],
    }

    (row,) = from_conversations([conversation], "tr", originals)

    assert row["id"] == "tr-0000" and row["augmented"] is augmented


def test_census_counts_each_fault_and_a_missing_prediction() -> None:
    rows = [_row("fg-1", SCALED), _row("fg-2", PLAIN), _row("fg-3", PLAIN)]
    predictions = {"fg-1": said(SCALED), "fg-2": said(PLAIN | {"year": 2025})}

    counts, each = census(rows, predictions)

    assert counts == {"no call when needed": 1, "year added": 1, "no prediction": 1}
    assert each == {
        "fg-1": ["no call when needed"],
        "fg-2": ["year added"],
        "fg-3": ["no prediction"],
    }


@pytest.mark.parametrize(
    ("calls", "needed"),
    [
        pytest.param(["100.00 / 100"], True, id="the label called, though 1 is printed"),
        pytest.param([], False, id="the label made no call"),
    ],
)
def test_a_row_with_the_labels_calls_needs_one_exactly_when_the_label_made_one(
    calls: list[str], needed: bool
) -> None:
    share = {"metric": "other", "value": 1.0, "unit": None, "year": None, "description": "held"}
    row = _row("va-1", share) | {"passage": "1 subsidiary, 100.00 held", "calls": calls}

    assert needs_call(cast(dict[str, JsonValue], row)) is needed
