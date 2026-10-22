"""The XBRL loader: periods from what a filing states, balances, segments, and what it refuses."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from aigent.facts.xbrl import Fact, label_of, parse

from .conftest import (
    DECEMBER,
    INTEGRATED_Q3,
    LEGACY_Q4,
    duration,
    filing,
    instant,
    nil,
    num,
    stated,
    text,
)

D = date.fromisoformat


@pytest.mark.parametrize(
    ("path", "concept", "start", "end", "segment", "value"),
    [
        pytest.param(
            LEGACY_Q4,
            "RevenueFromOperations",
            D("2023-04-01"),
            D("2024-03-31"),
            None,
            768404900000.0,
            id="a 2020-taxonomy year from its stated period, not its dates",
        ),
        pytest.param(
            LEGACY_Q4,
            "RevenueFromOperations",
            D("2024-01-01"),
            D("2024-03-31"),
            None,
            194464900000.0,
            id="a 2020-taxonomy quarter",
        ),
        pytest.param(
            LEGACY_Q4,
            "Assets",
            None,
            D("2024-03-31"),
            None,
            918261600000.0,
            id="a balance at the quarter's end",
        ),
        pytest.param(
            LEGACY_Q4,
            "CashAndCashEquivalentsCashFlowStatement",
            None,
            D("2023-03-31"),
            None,
            4059100000.0,
            id="opening cash, filed first, the day before the year",
        ),
        pytest.param(
            LEGACY_Q4,
            "CashAndCashEquivalentsCashFlowStatement",
            None,
            D("2024-03-31"),
            None,
            5965800000.0,
            id="closing cash, filed second, at the year's end",
        ),
        pytest.param(
            LEGACY_Q4,
            "SegmentRevenue",
            D("2023-04-01"),
            D("2024-03-31"),
            "Agri Business",
            161244300000.0,
            id="a segment named by its description",
        ),
        pytest.param(
            LEGACY_Q4,
            "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
            D("2023-04-01"),
            D("2024-03-31"),
            None,
            16.42,
            id="EPS per share",
        ),
        pytest.param(
            INTEGRATED_Q3,
            "RevenueFromOperations",
            D("2025-04-01"),
            D("2025-12-31"),
            None,
            660918500000.0,
            id="an integrated filing's nine months",
        ),
        pytest.param(
            INTEGRATED_Q3,
            "RevenueFromOperations",
            D("2025-10-01"),
            D("2025-12-31"),
            None,
            217066400000.0,
            id="an integrated filing's quarter",
        ),
        pytest.param(
            INTEGRATED_Q3,
            "AmountOfItemThatWillNotBeReclassifiedToProfitAndLoss",
            D("2025-04-01"),
            D("2025-12-31"),
            "Items that will not be reclassified to profit or loss",
            1465300000.0,
            id="a D_ context with no column, from its own dates",
        ),
    ],
)
def test_saved_filings_load_each_fact_with_its_period(
    path: Path, concept: str, start: date | None, end: date, segment: str | None, value: float
) -> None:
    facts = parse(path.read_bytes()).facts
    found = [
        f
        for f in facts
        if (f.concept, f.period_start, f.period_end, f.segment) == (concept, start, end, segment)
    ]
    assert [f.value for f in found] == [value], [
        (f.period_start, f.period_end) for f in facts if f.concept == concept
    ]
    assert found[0].period_kind == ("instant" if start is None else "duration")


@pytest.mark.parametrize("path", [LEGACY_Q4, INTEGRATED_Q3], ids=["2020 taxonomy", "integrated"])
def test_saved_filings_load_with_no_issues(path: Path) -> None:
    parsed = parse(path.read_bytes())
    assert parsed.issues == [], parsed.issues
    assert parsed.facts


def test_decimals_are_kept_and_inf_is_none() -> None:
    facts = parse(LEGACY_Q4.read_bytes()).facts
    by_concept = {f.concept: f.decimals for f in facts if f.period_start == D("2023-04-01")}
    assert by_concept["RevenueFromOperations"] == -7
    assert by_concept["BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations"] is None


@pytest.mark.parametrize(
    ("parts", "issue", "kept"),
    [
        pytest.param(
            (
                duration("FourD", "2025-04-01", "2025-12-31"),
                instant("OneI", "2025-12-31"),
                num("Revenue", "FourD", "10"),
                num("Assets", "OneI", "5"),
            ),
            "states no reporting period",
            ["Assets"],
            id="no stated period: durations dropped, balances kept",
        ),
        pytest.param(
            (*DECEMBER, duration("OneX", "2025-07-01", "2025-12-31"), num("Revenue", "OneX", "10")),
            "the stated period is used",
            ["Revenue"],
            id="a column dated otherwise than stated is read as stated",
        ),
        pytest.param(
            (*DECEMBER, num("Revenue", "OneD", "10"), num("Revenue", "OneD", "11")),
            "filed twice",
            ["Revenue"],
            id="one fact filed with two values keeps the first",
        ),
        pytest.param(
            (*DECEMBER, num("Revenue", "OneD", "ten")),
            "not a number",
            [],
            id="a value that is not a number",
        ),
        pytest.param(
            (*DECEMBER, num("Revenue", "Missing", "10")),
            "no context",
            [],
            id="a fact whose context is not filed",
        ),
        pytest.param(
            (
                duration("OneD", "2024-01-01", "2024-03-31"),
                duration("FourD", "2024-01-01", "2024-03-31"),
                stated("One", "2024-01-01", "2024-03-31"),
                stated("Four", "2023-04-01", "2024-03-31"),
                duration("D_X1", "2024-01-01", "2024-03-31"),
                num("Revenue", "D_X1", "10"),
            ),
            "no column, and the dates do not hold",
            [],
            id="no column in a filing whose dates do not hold",
        ),
        pytest.param(
            (
                duration("OneD", "2024-01-01", "2024-03-31"),
                stated("One", "2024-01-01", "2024-03-31"),
                stated("Four", "2023-04-01", "2024-03-31"),
                duration("D_Cash", "2024-01-01", "2024-03-31"),
                num("CashAndCashEquivalentsCashFlowStatement", "D_Cash", "10"),
            ),
            "a balance with no stated period",
            [],
            id="opening or closing cash in a context with no column",
        ),
    ],
)
def test_what_cannot_be_settled_is_an_issue_not_a_fact(
    parts: tuple[str, ...], issue: str, kept: list[str]
) -> None:
    parsed = parse(filing(*parts))
    assert any(issue in i for i in parsed.issues), parsed.issues
    assert [f.concept for f in parsed.facts] == kept


def test_a_column_dated_otherwise_takes_the_stated_period() -> None:
    parsed = parse(
        filing(
            *DECEMBER, duration("OneX", "2025-07-01", "2025-12-31"), num("Revenue", "OneX", "10")
        )
    )
    assert (parsed.facts[0].period_start, parsed.facts[0].period_end) == (
        D("2025-10-01"),
        D("2025-12-31"),
    )


def test_the_first_of_two_values_is_kept() -> None:
    parsed = parse(filing(*DECEMBER, num("Revenue", "OneD", "10"), num("Revenue", "OneD", "11")))
    assert [f.value for f in parsed.facts] == [10.0]


@pytest.mark.parametrize(
    ("parts", "facts", "skipped"),
    [
        pytest.param(
            (
                *DECEMBER,
                instant("I_RelatedPartyTransaction01", "2025-12-31"),
                num("AmountOfTransaction", "I_RelatedPartyTransaction01", "10"),
            ),
            0,
            1,
            id="a related-party transaction is out of scope",
        ),
        pytest.param((*DECEMBER, nil("Revenue", "OneD")), 0, 0, id="a nil value"),
        pytest.param((*DECEMBER, num("Revenue", "OneD", "")), 0, 0, id="an empty value"),
        pytest.param(
            (*DECEMBER, num("Revenue", "OneD", "10"), num("Revenue", "OneD", "10")),
            1,
            0,
            id="the same fact filed twice is one",
        ),
        pytest.param(
            (*DECEMBER, text("NameOfTheCompany", "OneD", "T Ltd")),
            0,
            0,
            id="a text fact is not a fact",
        ),
    ],
)
def test_what_is_left_out_quietly(parts: tuple[str, ...], facts: int, skipped: int) -> None:
    parsed = parse(filing(*parts))
    assert (len(parsed.facts), parsed.skipped, parsed.issues) == (facts, skipped, [])


def test_a_segment_with_no_description_is_named_by_its_member() -> None:
    parsed = parse(
        filing(
            *DECEMBER,
            duration("OneSeg01D", "2025-10-01", "2025-12-31", member="Segment01Member"),
            num("SegmentRevenue", "OneSeg01D", "10"),
        )
    )
    assert [f.segment for f in parsed.facts] == ["Segment01Member"]


def test_a_fact_carries_its_value_and_unit() -> None:
    parsed = parse(filing(*DECEMBER, num("BasicEps", "OneD", "3.94", "INRPerShare", "INF")))
    assert parsed.facts == [
        Fact(
            "BasicEps",
            "duration",
            D("2025-10-01"),
            D("2025-12-31"),
            None,
            3.94,
            "INRPerShare",
            None,
        )
    ]


@pytest.mark.parametrize(
    ("concept", "label"),
    [
        ("RevenueFromOperations", "Revenue from operations"),
        (
            "ProfitOrLossAttributableToOwnersOfParent",
            "Profit or loss attributable to owners of parent",
        ),
        ("NoncurrentAssets", "Noncurrent assets"),
        ("Revenue01", "Revenue 01"),
    ],
)
def test_label_of_reads_an_element_name_as_words(concept: str, label: str) -> None:
    assert label_of(concept) == label


def test_a_first_quarter_states_one_column_and_its_dates_hold() -> None:
    parsed = parse(
        filing(
            duration("OneD", "2025-04-01", "2025-06-30"),
            stated("One", "2025-04-01", "2025-06-30"),
            duration("D_Items11", "2025-04-01", "2025-06-30"),
            num("Revenue", "OneD", "10"),
            num("AmountOfItem", "D_Items11", "2"),
        )
    )
    assert parsed.issues == []
    assert [(f.concept, f.period_start) for f in parsed.facts] == [
        ("Revenue", D("2025-04-01")),
        ("AmountOfItem", D("2025-04-01")),
    ]


def test_a_name_filed_with_a_unit_is_still_a_name() -> None:
    parsed = parse(
        filing(
            *DECEMBER,
            duration("OneSeg01D", "2025-10-01", "2025-12-31", member="Segment01Member"),
            num("DescriptionOfReportableSegment", "OneSeg01D", "Treasury"),
            num("SegmentRevenue", "OneSeg01D", "10"),
        )
    )
    assert parsed.issues == []
    assert [(f.concept, f.segment) for f in parsed.facts] == [("SegmentRevenue", "Treasury")]
