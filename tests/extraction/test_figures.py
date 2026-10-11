"""The figure schema's prompt, its parser, and the grader the fine-tune is scored by."""

from __future__ import annotations

import json
import re

import pytest
from pydantic import JsonValue

from aigent.evals.dataset import Case, load_jsonl
from aigent.evals.grade import Outcome
from aigent.extraction.figures import (
    DATASET,
    METRICS,
    SYSTEM,
    TOOL,
    TOOL_RULE,
    Figure,
    Figures,
    Labelled,
    check_call,
    figure_messages,
    figures_match,
    parse_figures,
    transcript,
    unit_key,
)
from aigent.retrieval.chunk import squeeze
from aigent.retrieval.questions import DATASET as RETRIEVAL

STATED = "stated in the passage"


def fig(
    value: float,
    unit: str | None = None,
    year: int | None = None,
    metric: str = "other",
    description: str | None = STATED,
) -> JsonValue:
    """A figure as JSON; an other figure is described by default, and no other metric is."""
    said = description if metric == "other" or description != STATED else None
    return {"metric": metric, "value": value, "unit": unit, "year": year, "description": said}


def grade(
    want: list[JsonValue], got: JsonValue, passage: str | None = None
) -> tuple[bool, str, dict[str, bool]]:
    given: dict[str, JsonValue] = {} if passage is None else {"passage": passage}
    case = Case(id="fg-x", input=given, expected={"figures": want})
    score = figures_match(case, Outcome(output={"figures": got}))
    return score.passed, score.detail, dict(score.parts)


def test_the_prompt_carries_every_field_rule_and_the_inputs() -> None:
    system, user = figure_messages("Exports to over 95 countries", "ITC FY25", "export reach")

    assert system == SYSTEM
    for name, field in Figure.model_fields.items():
        assert field.description and field.description in system, f"{name} rule missing"
    for metric in METRICS:
        assert metric in system, f"{metric} is not offered to the model"
    for part in ("Exports to over 95 countries", "ITC FY25", "export reach"):
        assert part in user


@pytest.mark.parametrize(
    ("unit", "key"),
    [
        pytest.param("INR", "INR", id="a currency code"),
        pytest.param("Rs.", "INR", id="Rs is rupees"),
        pytest.param("₹", "INR", id="the rupee sign"),
        pytest.param("usd", "USD", id="a code in lower case"),
        pytest.param(None, None, id="no unit"),
        pytest.param("  ", None, id="a blank unit"),
        pytest.param("count", "count", id="anything else as given, so it cannot pass for null"),
    ],
)
def test_unit_key(unit: str | None, key: str | None) -> None:
    assert unit_key(unit) == key


@pytest.mark.parametrize(
    ("want", "got", "passed", "detail"),
    [
        pytest.param([fig(3000000, "INR")], [fig(3000000, "Rs")], True, "", id="Rs is INR"),
        pytest.param(
            [fig(3000000, "INR")], [fig(3000000, None)], False, "missed", id="the currency dropped"
        ),
        pytest.param(
            [fig(3000000, "INR")], [fig(30, "INR")], False, "missed", id="a lakh left unapplied"
        ),
        pytest.param(
            [fig(0.45, None, 2025)],
            [fig(45, None, 2025)],
            False,
            "missed",
            id="a percentage not made a fraction",
        ),
        pytest.param(
            [fig(95, description="countries")],
            [fig(95, description="export countries")],
            True,
            "",
            id="a description need not match the label",
        ),
        pytest.param([fig(95)], [fig(70)], False, "missed other 95", id="a distractor"),
        pytest.param(
            [fig(0.45, None, 2025)],
            [fig(0.45, None, None)],
            False,
            "missed",
            id="the year dropped",
        ),
        pytest.param(
            [fig(0.022, None, 2024)],
            [fig(0.022, None, 2024), fig(185000000, None, 2024)],
            False,
            "extra other 1.85e+08",
            id="a figure not asked for",
        ),
        pytest.param(
            [fig(0.3, None, 2024), fig(0.45, None, 2025)],
            [fig(0.45, None, 2025), fig(0.3, None, 2024)],
            True,
            "",
            id="both figures in any order",
        ),
        pytest.param(
            [fig(2860000, "AUD", 2025, "revenue")],
            [fig(2860000, "AUD", 2025, "profit")],
            False,
            "missed revenue",
            id="the wrong metric",
        ),
        pytest.param([], [], True, "", id="nothing wanted, nothing given"),
        pytest.param([fig(1)], "not a list", False, "not figures", id="malformed output"),
    ],
)
def test_figures_match(want: list[JsonValue], got: JsonValue, passed: bool, detail: str) -> None:
    ok, said, _ = grade(want, got)

    assert ok is passed, said
    assert detail in said, said


def test_figures_match_reports_what_was_missed_and_what_was_extra_apart() -> None:
    _, _, parts = grade([fig(1), fig(2)], [fig(1), fig(9)])

    assert parts == {"all_found": False, "no_extra": False, "described": True}


PASSAGE = "Pioneer in farmer empowerment F Exports to over 95 countries F ITCMAARS covers FPOs"


@pytest.mark.parametrize(
    ("figure", "passage", "described"),
    [
        pytest.param(
            fig(95, description="Exports to over 95 countries"),
            PASSAGE,
            True,
            id="other, verbatim in the passage",
        ),
        pytest.param(
            fig(95, description="EXPORTS to  over 95 countries"),
            PASSAGE,
            True,
            id="verbatim but for case and spacing",
        ),
        pytest.param(
            fig(95, description="Exported to 95 nations"),
            PASSAGE,
            False,
            id="other, paraphrased",
        ),
        pytest.param(fig(95, description=None), PASSAGE, False, id="other with no description"),
        pytest.param(fig(95, description=""), None, False, id="other with an empty description"),
        pytest.param(
            fig(95, description="any words"),
            None,
            True,
            id="no passage to check against",
        ),
        pytest.param(
            fig(2860000, "AUD", metric="revenue", description="turnover"),
            None,
            False,
            id="a named metric described anyway",
        ),
        pytest.param(
            fig(2860000, "AUD", metric="revenue"),
            None,
            True,
            id="a named metric left undescribed",
        ),
    ],
)
def test_a_figure_is_described_exactly_when_its_metric_is_other(
    figure: JsonValue, passage: str | None, described: bool
) -> None:
    ok, said, parts = grade([figure], [figure], passage)

    assert parts["described"] is described, said
    assert ok is described, said


@pytest.mark.parametrize(
    ("text", "values"),
    [
        pytest.param('{"figures": []}', [], id="a bare object"),
        pytest.param(
            'Here:\n```json\n{"figures": [{"metric": "m", "value": 4, "unit": "share", '
            '"year": null}]}\n```',
            [4.0],
            id="fenced with chatter",
        ),
        pytest.param("no json here", None, id="no object"),
        pytest.param('{"figures": [{"value": 4}]}', None, id="a record missing fields"),
        pytest.param('{"figures": [', None, id="truncated"),
    ],
)
def test_parse_figures(text: str, values: list[float] | None) -> None:
    parsed = parse_figures(text)

    assert (None if parsed is None else [f.value for f in parsed.figures]) == values


def test_every_test_case_names_a_chunk_and_holds_valid_figures() -> None:
    cases = load_jsonl(DATASET)

    assert cases, "the test set is empty"
    for case in cases:
        assert str(case.input["chunk_id"]).count("#") == 1, case.id
        assert case.input["wanted"], case.id
        figures = Figures.model_validate(case.expected).figures
        assert figures, case.id
        assert all(f.metric in METRICS for f in figures), case.id
        assert all((f.metric == "other") == bool(f.description) for f in figures), case.id


YEAR = re.compile(r"(19|20)\d\d")


def _runs(text: str, n: int = 4) -> set[str]:
    words = squeeze(text).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def test_no_example_in_the_prompt_comes_from_the_test_set() -> None:
    """An example that is also a test case hands the model that row; it still counts as a pass."""
    quotes = {case.id: str(case.expected["quote"]) for case in load_jsonl(RETRIEVAL)}
    numbers = re.compile(r"\d[\d,.]*\d")  # a year like 2024 is in any fiscal prompt
    leaked: dict[str, list[str]] = {}
    for case in load_jsonl(DATASET):
        quote = quotes[next(t for t in case.tags if t.startswith("from-")).removeprefix("from-")]
        shared = sorted(_runs(quote) & _runs(SYSTEM))
        shared += [n for n in numbers.findall(quote) if len(n) >= 3 and n in SYSTEM]
        shared = [n for n in shared if not YEAR.fullmatch(n)]
        if shared:
            leaked[case.id] = shared
    assert not leaked, f"the prompt quotes the test set: {leaked}"


TABLE = "Revenue (Rs in crore) 2,871.59 1,665.71; margin 7.25%; staff 1,234; (1,554.3) loss; ten"


def labelled(
    value: float,
    call: str | None,
    metric: str = "revenue",
    unit: str | None = "INR",
    description: str | None = None,
) -> Labelled:
    return Labelled(
        metric=metric, value=value, unit=unit, year=2025, description=description, call=call
    )


@pytest.mark.parametrize(
    ("label", "ok"),
    [
        pytest.param(labelled(28715900000, "2871.59 * 10**7"), True, id="a heading's crore"),
        pytest.param(labelled(0.0725, "7.25 / 100", "margin", None), True, id="a percentage"),
        pytest.param(labelled(-15543000000, "-1554.3 * 10**7"), True, id="parentheses, negative"),
        pytest.param(labelled(1234, None, "employees", None), True, id="printed in full"),
        pytest.param(labelled(10, None, "other", None, "ten"), True, id="a number word"),
        pytest.param(labelled(2871590000, "2871.59 * 10**7"), False, id="a call it does not give"),
        pytest.param(labelled(2871590, "2871.59 * 10**3"), True, id="thousands"),
        pytest.param(labelled(28715900, "2871.59 * 10**4"), False, id="no such scale"),
        pytest.param(labelled(99990000000, "9999 * 10**7"), False, id="a number not printed"),
        pytest.param(labelled(2871.59, None), True, id="the bare cell, unscaled"),
        pytest.param(labelled(28715900000, None), False, id="scaled with no call"),
        pytest.param(labelled(28715900000, "2871.59*10**7"), False, id="a call not in the form"),
        pytest.param(labelled(5, "5 + 0"), False, id="arithmetic that is not scaling"),
    ],
)
def test_check_call(label: Labelled, ok: bool) -> None:
    assert check_call(label, TABLE) is ok


def test_a_transcript_makes_every_call_in_one_turn_then_answers() -> None:
    labels = [
        labelled(28715900000, "2871.59 * 10**7"),
        labelled(0.0725, "7.25 / 100", "margin", None),
        labelled(1234, None, "employees", None),
    ]

    messages = transcript(TABLE, "source", "wanted", labels)

    assert messages is not None
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"]
    sent = json.loads(json.dumps(messages[2]["tool_calls"]))
    assert [c["function"]["arguments"]["expression"] for c in sent] == [
        "2871.59 * 10**7",
        "7.25 / 100",
    ]
    assert [m["content"] for m in messages[3:5]] == ["28715900000.0", "0.0725"]
    final = Figures.model_validate(json.loads(str(messages[-1]["content"]))).figures
    assert [f.value for f in final] == [28715900000, 0.0725, 1234]
    assert "call" not in str(messages[-1]["content"]), "the teacher's call is not in the answer"


def test_a_transcript_needs_no_tool_turn_when_every_number_is_printed_in_full() -> None:
    messages = transcript(TABLE, "s", "w", [labelled(1234, None, "employees", None)])

    assert messages is not None
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(labelled(2871590000, "2871.59 * 10**7"), id="a call that fails its check"),
        pytest.param(
            labelled(1234, None, "other", None, "staff headcount"), id="a paraphrased description"
        ),
        pytest.param(labelled(1234, None, "deposit", None), id="a metric not in the vocabulary"),
    ],
)
def test_a_label_that_fails_a_check_teaches_nothing(label: Labelled) -> None:
    assert transcript(TABLE, "s", "w", [labelled(1234, None, "employees", None), label]) is None


def test_the_prompt_tells_the_model_to_leave_arithmetic_to_the_tool() -> None:
    assert TOOL_RULE in SYSTEM
    function = TOOL["function"]
    assert isinstance(function, dict) and function["name"] == "calculate"
