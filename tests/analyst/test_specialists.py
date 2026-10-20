"""What the report analyst's check accepts and refuses."""

from __future__ import annotations

import pytest

from aigent.analyst.specialists import (
    FigureJob,
    Invalid,
    Passage,
    Told,
    canonical,
    check_figures,
)

JOB = FigureJob("What revenue and EBITDA margin did ITC report for FY25?", "ITC", 2025)
READ = Passage(
    "ITC-FY25#0007",
    "ITC-FY25 p12",
    "Revenue from operations stood at ₹ 2,345 crore; the EBITDA margin was 18%.",
)
OTHER = Passage("ITC-FY25#0090", "ITC-FY25 p80", "The Board met eight times during the year.")


def told(text: str, *calls: str) -> Told:
    return Told(text, 0.0, 0, 0, calls, (READ, OTHER))


def figures(*rows: tuple[str, float], cited: str = READ.id) -> str:
    made = [
        f'{{"metric": "{m}", "value": {v}, "unit": null, "year": 2025, "description": null, '
        f'"chunk_id": "{cited}"}}'
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
        pytest.param(figures(), (), [], id="the report states none of it"),
        pytest.param(
            figures(("revenue", 23450000000.0)),
            ("(2345 * 10000000)",),
            [23450000000.0],
            id="a scale written out in full",
        ),
    ],
)
def test_figures_whose_every_value_is_their_passage_s_own_pass(
    reply: str, calls: tuple[str, ...], values: list[float]
) -> None:
    found = check_figures(JOB, told(reply, *calls))

    assert [f.value for f in found.figures] == values
    assert [f.chunk_id for f in found.figures] == [READ.id] * len(values)
    assert found.passages == ({READ.id: READ} if values else {}), "only the passages cited"


@pytest.mark.parametrize(
    ("reply", "calls", "reason"),
    [
        pytest.param("no figures here", (), "no figures", id="no JSON"),
        pytest.param(figures(("revenue", 2346.0)), (), "2346", id="a number not printed"),
        pytest.param(
            figures(("other", 2345.0), cited=OTHER.id),
            (),
            "ITC-FY25#0090 does not back 2345",
            id="a number printed, but in another passage than the one cited",
        ),
        pytest.param(
            figures(("other", 2345.0), cited="ITC-FY25#0500"),
            (),
            "ITC-FY25#0500 is not a passage its searches returned",
            id="a passage no search returned",
        ),
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


SEGMENT = Passage(
    "ITC-FY25#0300",
    "ITC-FY25 p90",
    "Segment Revenue: FMCG - Others 2,345 crore; Agri Business 900.",
)


@pytest.mark.parametrize(
    ("basis", "passes"),
    [
        pytest.param(None, True, id="no basis"),
        pytest.param("consolidated", True, id="consolidated"),
        pytest.param("standalone", True, id="standalone"),
        pytest.param("FMCG - Others", True, id="a segment the passage names"),
        pytest.param("Cigarettes", False, id="a segment it does not"),
    ],
)
def test_a_figure_s_basis_is_a_statement_or_a_segment_its_passage_names(
    basis: str | None, passes: bool
) -> None:
    named = "null" if basis is None else f'"{basis}"'
    reply = (
        '{"figures": [{"metric": "revenue", "value": 2345.0, "unit": null, "year": 2025, '
        f'"description": null, "basis": {named}, "chunk_id": "{SEGMENT.id}"}}]}}'
    )
    said = Told(reply, 0.0, 0, 0, (), (SEGMENT,))

    if passes:
        assert check_figures(JOB, said).figures[0].basis == basis
    else:
        with pytest.raises(Invalid, match="does not name the segment 'Cigarettes'"):
            check_figures(JOB, said)
