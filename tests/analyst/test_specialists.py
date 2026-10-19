"""What the report analyst's check accepts and refuses."""

from __future__ import annotations

import pytest

from aigent.analyst.specialists import FigureJob, Invalid, Told, canonical, check_figures


def told(text: str, *calls: str) -> Told:
    return Told(text, 0.0, 0, 0, calls)


JOB = FigureJob(
    passage="Revenue from operations stood at ₹ 2,345 crore; the EBITDA margin was 18%.",
    source="ITC-FY25",
    wanted="revenue and margin",
)


def figures(*rows: tuple[str, float]) -> str:
    made = [
        f'{{"metric": "{m}", "value": {v}, "unit": null, "year": 2025, "description": null}}'
        for m, v in rows
    ]
    return '{"figures": [' + ", ".join(made) + "]}"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("330943 * 10**7", "330943 * 10**7"),
        ("330943 * 10000000", "330943 * 10**7"),
        ("330943*1e7", "330943 * 10**7"),
        ("(330943 * 10000000)", "330943 * 10**7"),
        ("-12.5 * 1000", "-12.5 * 10**3"),
        pytest.param("330943 * 100", "330943 * 100", id="a hundred is not a scale"),
        pytest.param("330943 / 1.079", "330943 / 1.079", id="arithmetic is left alone"),
        pytest.param("7.9 / 100", "7.9 / 100", id="a percentage is left alone"),
    ],
)
def test_a_scaling_call_is_read_in_its_one_form(call: str, expected: str) -> None:
    assert canonical(call) == expected


@pytest.mark.parametrize(
    ("reply", "calls", "values"),
    [
        pytest.param(figures(("other", 2345.0)), (), [2345.0], id="a number as printed"),
        pytest.param(
            figures(("revenue", 23450000000.0), ("margin", 0.18)),
            ("2345 * 10**7", "18 / 100"),
            [23450000000.0, 0.18],
            id="a named scale and a percentage, each through a call",
        ),
        pytest.param(figures(), (), [], id="nothing wanted is in the passage"),
        pytest.param(
            figures(("revenue", 23450000000.0)),
            ("(2345 * 10000000)",),
            [23450000000.0],
            id="a scale written out in full",
        ),
    ],
)
def test_figures_whose_every_value_is_the_passage_s_own_pass(
    reply: str, calls: tuple[str, ...], values: list[float]
) -> None:
    assert [f.value for f in check_figures(JOB, told(reply, *calls)).figures] == values


@pytest.mark.parametrize(
    ("reply", "calls", "reason"),
    [
        pytest.param("no figures here", (), "no figures", id="no JSON"),
        pytest.param(figures(("revenue", 2346.0)), (), "2346", id="a number not printed"),
        pytest.param(
            figures(("revenue", 23450000000.0)), (), "23450000000", id="a scale with no call"
        ),
        pytest.param(
            figures(("revenue", 2345000000.0)),
            ("2345 * 10**6",),
            "2345000000",
            id="a scale the passage does not name",
        ),
        pytest.param(
            figures(("revenue", 23450000000.0), ("margin", 0.19)),
            ("2345 * 10**7", "18 / 100"),
            "0.19",
            id="one bad figure refuses them all",
        ),
    ],
)
def test_figures_with_a_value_the_passage_does_not_back_are_refused(
    reply: str, calls: tuple[str, ...], reason: str
) -> None:
    with pytest.raises(Invalid, match=reason):
        check_figures(JOB, told(reply, *calls))
