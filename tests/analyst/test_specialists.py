"""What each specialist's check accepts and refuses."""

from __future__ import annotations

import pytest

from aigent.analyst.specialists import (
    FigureJob,
    Invalid,
    Told,
    canonical,
    check_figures,
    check_headline,
)
from aigent.extraction.headlines import Resolver

RECORD = (
    '{"company": "Infy", "metric": "profit", "quarter": "Q2", "direction": "up", '
    '"change_pct": 11.0}'
)
HEADLINE = "Infy Q2 profit rises 11% on deal wins"


def told(text: str, *calls: str) -> Told:
    return Told(text, 0.0, 0, 0, calls)


def test_a_headline_record_that_passes_carries_its_ticker() -> None:
    record = check_headline(Resolver())(HEADLINE, told(f"Here it is: {RECORD}"))

    assert record["ticker"] == "INFY" and record["change_pct"] == 11.0, record


def test_a_headline_naming_no_company_passes_with_no_ticker() -> None:
    sector = RECORD.replace('"Infy"', "null")

    assert (
        check_headline(Resolver())("IT sector Q2 profit rises 11%", told(sector))["ticker"] is None
    )


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        pytest.param("I could not tell.", "no headline record", id="no JSON"),
        pytest.param(RECORD.replace("profit", "growth"), "metric 'growth'", id="a metric made up"),
        pytest.param(RECORD.replace("Q2", "Q5"), "quarter 'Q5'", id="a quarter made up"),
        pytest.param(RECORD.replace('"up"', '"rising"'), "direction 'rising'", id="a direction"),
        pytest.param(RECORD.replace("11.0", "12.5"), "12.5", id="a percentage not printed"),
        pytest.param(RECORD.replace("Infy", "Infoworks"), "'Infoworks'", id="an unknown company"),
    ],
)
def test_a_headline_record_that_cannot_be_trusted_is_refused(reply: str, reason: str) -> None:
    with pytest.raises(Invalid, match=reason):
        check_headline(Resolver())(HEADLINE, told(reply))


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
