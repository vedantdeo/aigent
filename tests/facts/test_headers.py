"""`facts.headers`: each column's period from a table's header, and a document's scale."""

from __future__ import annotations

from datetime import date

import pytest

from aigent.facts.headers import Column, column_periods, document_scale

from .conftest import CROSSCHECK_HEADER as _HEADER


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        pytest.param(
            (
                ("Six Month ended Year ended",),
                ("", "Three months ended", "", "Six Month ended", "Year ended"),
                ("", "Sep 30, 2025", "Jun 30, 2025", "Sep 30, 2025", "Mar 31, 2025"),
            ),
            {
                1: Column(date(2025, 9, 30), 3),
                2: Column(date(2025, 6, 30), 3),
                3: Column(date(2025, 9, 30), 6),
                4: Column(date(2025, 3, 31), 12),
            },
            id="Wipro's stray heading line, the nearest heading over the dates winning",
        ),
        pytest.param(
            _HEADER,
            {
                2: Column(date(2024, 6, 30), 3),
                3: Column(date(2024, 3, 31), 3),
                4: Column(date(2023, 6, 30), 3),
                5: Column(date(2024, 3, 31), 12),
            },
            id="a group heading over dated columns",
        ),
        pytest.param(
            (
                ("Six months ended",),
                ("", "Three months ended", "", "Six months ended", "Year ended"),
                (
                    "Particulars",
                    "September 30, 2025 (Q2-2026)",
                    "June 30, 2025 (Q1-2026)",
                    "September 30, 2025 (H1-2026)",
                    "March 31, 2025 (FY2025)",
                ),
            ),
            {
                1: Column(date(2025, 9, 30), 3),
                2: Column(date(2025, 6, 30), 3),
                3: Column(date(2025, 9, 30), 6),
                4: Column(date(2025, 3, 31), 12),
            },
            id="a column's own cells over a stray heading",
        ),
        pytest.param(
            (("Particulars", "As at 31.03.2025", "Asat 31.03.2024"),),
            {1: Column(date(2025, 3, 31), 0), 2: Column(date(2024, 3, 31), 0)},
            id="balances as at a day",
        ),
        pytest.param(
            (("", "Nine months ended", ""), ("Particulars", "31.12.2024", "31.12.2023")),
            {1: Column(date(2024, 12, 31), 9), 2: Column(date(2023, 12, 31), 9)},
            id="nine months",
        ),
        pytest.param(
            (("Particulars", "31.12.2024"),),
            {1: Column(date(2024, 12, 31), None)},
            id="a date with no length",
        ),
        pytest.param(
            (
                ("Particulars", "Quarter Ended", "", "", "Half Year Ended"),
                ("", "30 Sep'24", "30 Jun'24", "30 Sep'23", "30 Sep'24"),
            ),
            {
                1: Column(date(2024, 9, 30), 3),
                2: Column(date(2024, 6, 30), 3),
                3: Column(date(2023, 9, 30), 3),
                4: Column(date(2024, 9, 30), 6),
            },
            id="Reliance's day, month and two-digit year",
        ),
        pytest.param(
            (("", "Quarter Ended", "", ""), ("", "3osea'24", "30 Sen'23", "30 Seo'24")),
            {
                1: Column(date(2024, 9, 30), 3),
                2: Column(date(2023, 9, 30), 3),
                3: Column(date(2024, 9, 30), 3),
            },
            id="an OCR'd month read by the two letters only September starts with",
        ),
        pytest.param(
            (("", "Quarter Ended", ""), ("", "30 Jnu'24", "31 Mav'24")),
            {},
            id="an OCR'd month two months start like is left undated",
        ),
        pytest.param(
            (
                ("Particulars", "Quarter Ended", "", "", "Half Year Ended", "", "Year Ended"),
                (
                    "",
                    "30thSeo'25",
                    "30 th Jun'25",
                    "30 th Sep'24",
                    "30th Sep'25",
                    "30th Seo'24",
                    "31 st Mar'25",
                ),
            ),
            {
                1: Column(date(2025, 9, 30), 3),
                2: Column(date(2025, 6, 30), 3),
                3: Column(date(2024, 9, 30), 3),
                4: Column(date(2025, 9, 30), 6),
                5: Column(date(2024, 9, 30), 6),
                6: Column(date(2025, 3, 31), 12),
            },
            id="Reliance's ordinal days",
        ),
        pytest.param(
            (
                ("", "Quarter Ended", "", "Half Year Ended"),
                ("Particulars", "30.09.2025", "30.06.2025", "30.09.2025"),
                (
                    "Earnings per equity share (Not Annualised for the quarter/half year)",
                    "",
                    "",
                    "",
                ),
            ),
            {
                1: Column(date(2025, 9, 30), 3),
                2: Column(date(2025, 6, 30), 3),
                3: Column(date(2025, 9, 30), 6),
            },
            id="a title in the label column carries over no column",
        ),
        pytest.param(
            (("", "", "(₹ in crores)", ""), ("", "PROFITS", "2024 - 25", "2023 - 24")),
            {2: Column(date(2025, 3, 31), 12), 3: Column(date(2024, 3, 31), 12)},
            id="ITC's financial years",
        ),
        pytest.param(
            (("Particulars", "FY25", "FY 2024"),),
            {1: Column(date(2025, 3, 31), 12), 2: Column(date(2024, 3, 31), 12)},
            id="FY short forms",
        ),
        pytest.param(
            (("Particulars", "2019-24"),),
            {},
            id="a span of years is no financial year",
        ),
        pytest.param(
            (
                ("", "", "For the year ended", ""),
                ("Particulars", "Note No.", "2025", "(₹ In Crore) 31 March 2024"),
            ),
            {2: Column(date(2025, 3, 31), 12), 3: Column(date(2024, 3, 31), 12)},
            id="Bajaj Auto's bare year takes its neighbour's day",
        ),
        pytest.param(
            (("Particulars", "2025", "2024"),),
            {},
            id="bare years with no dated neighbour stay undated",
        ),
        pytest.param(
            (("", "Year ended", "", ""), ("Particulars", "2025", "31 March 2024", "31.12.2023")),
            {2: Column(date(2024, 3, 31), 12), 3: Column(date(2023, 12, 31), 12)},
            id="a bare year among neighbours of different days stays undated",
        ),
        pytest.param(
            (("Particulars", "Year ended 31st March, 2025", "As at 31 st March 2024"),),
            {1: Column(date(2025, 3, 31), 12), 2: Column(date(2024, 3, 31), 0)},
            id="the day first, as annual reports print it",
        ),
        pytest.param(
            (("Particulars", "December 2023", "31.03.2024"),),
            {2: Column(date(2024, 3, 31), None)},
            id="a month and year with no day takes no neighbour's day",
        ),
        pytest.param(
            (
                ("(₹ in crores)",),
                ("2024",),
                ("", "", "As at March 31, 2025", "As at March 31,"),
                ("", "", "2024", ""),
            ),
            {2: Column(date(2025, 3, 31), 0)},
            id="Tata Motors' year on a line of its own is a wrapped title, not a column",
        ),
        pytest.param(
            (
                ("", "Year ended March 31,", "", "Quarter ended June 30,"),
                ("Particulars", "2024", "2025", "2025"),
            ),
            {
                1: Column(date(2024, 3, 31), 12),
                2: Column(date(2025, 3, 31), 12),
                3: Column(date(2025, 6, 30), 3),
            },
            id="Wipro's years under a heading naming the day",
        ),
        pytest.param(
            (("", "September 30,", "March 31,"), ("Particulars", "2025", "2025*")),
            {1: Column(date(2025, 9, 30), None), 2: Column(date(2025, 3, 31), None)},
            id="a year marked restated with an asterisk",
        ),
        pytest.param(
            (
                (
                    "Particulars",
                    "3 Months ended 31.12.2023",
                    "Preceding 3 Month\ufffd ended 30.09.2023",
                    "9 Month\u2022 ended 31.12.2022",
                    "Twelve Months ended 31.03.2023",
                    "12 Months ended 31.03.2024",
                ),
            ),
            {
                1: Column(date(2023, 12, 31), 3),
                2: Column(date(2023, 9, 30), 3),
                3: Column(date(2022, 12, 31), 9),
                4: Column(date(2023, 3, 31), 12),
                5: Column(date(2024, 3, 31), 12),
            },
            id="ITC's months in figures, one OCR'd with a stray mark",
        ),
        pytest.param(
            (
                ("STATEMENT FOR THE QUARTER AND NINE MONTHS ENDED DECEMBER 31, 2024",),
                ("", "", "", "Quarter ended", "", "Nine months", "ended", "Year ended"),
                ("", "Particulars", "Dec 31, 2024", "Sep 30, 2024", "Dec 31, 2023", "Dec 31, 2024")
                + ("Dec 31, 2023", "Mar 31, 2024"),
            ),
            {
                2: Column(date(2024, 12, 31), 3),
                3: Column(date(2024, 9, 30), 3),
                4: Column(date(2023, 12, 31), 3),
                5: Column(date(2024, 12, 31), 9),
                6: Column(date(2023, 12, 31), 9),
                7: Column(date(2024, 3, 31), 12),
            },
            id="Tata's title carried nowhere and its quarter heading over the middle column",
        ),
        pytest.param(
            (
                (
                    "",
                    "Quarter ended 30.06.2024",
                    "31.03.2024",
                    "30.06.2023",
                    "Year ended 31.03.2024",
                ),
            ),
            {
                1: Column(date(2024, 6, 30), 3),
                2: Column(date(2024, 3, 31), 3),
                3: Column(date(2023, 6, 30), 3),
                4: Column(date(2024, 3, 31), 12),
            },
            id="a group's length runs along it, not in from the next group",
        ),
        pytest.param(
            (
                ("", "", "", "", "", "Year ended"),
                ("", "Particulars", "30.06.2024", "Quarter ended 31.03.2024", "30.06.2023")
                + ("31.03.2024",),
            ),
            {
                2: Column(date(2024, 6, 30), 3),
                3: Column(date(2024, 3, 31), 3),
                4: Column(date(2023, 6, 30), 3),
                5: Column(date(2024, 3, 31), 12),
            },
            id="Bajaj's quarter heading beside the column it also heads",
        ),
        pytest.param(
            (
                ("STANDALONE RESULTS FOR THE QUARTER AND SIX MONTHS ENDED SEPTEMBER 30, 2025",),
                ("", "", "September 30,", "Quarter ended June 30,", "September 30,")
                + ("Sil< months September 30,", "ended Sectember 30,", "Year ended March 31,"),
                ("Particulars", "", "2025", "202s•", "2024•", "2025", "2024°", "2025•"),
                ("", "", "Audited", "# (Audited", "Unaudited", "Audited", "Unaudited", "Audited"),
            ),
            {
                2: Column(date(2025, 9, 30), 3),
                3: Column(date(2025, 6, 30), 3),
                4: Column(date(2024, 9, 30), 3),
                5: Column(date(2025, 9, 30), 6),
                6: Column(date(2024, 9, 30), 6),
                7: Column(date(2025, 3, 31), 12),
            },
            id="Tata's OCR'd header, each column's missing parts from the title",
        ),
        pytest.param(
            (
                ("Results for the Quarter and Twelve Months ended 31st March, 2024",),
                ("Particulars", "#", "3 Months ended 31.12.2023", "Twelve Months ended"),
                ("", "(Audited", "(Unaudited)", ""),
            ),
            {
                1: Column(date(2024, 3, 31), 3),
                2: Column(date(2023, 12, 31), 3),
                3: Column(date(2024, 3, 31), 12),
            },
            id="ITC's columns that lost their dates take the title's",
        ),
        pytest.param(
            (("", "As at current half year end", "As at previous year end"),)
            + (("", "30.09.2024", "31.03.2024"),),
            {1: Column(date(2024, 9, 30), 0), 2: Column(date(2024, 3, 31), 0)},
            id="ITC's balance at a half year's end",
        ),
        pytest.param(
            (
                ("", "Six.months ended", "", "Nine-months ended"),
                ("", "31 .12.2022", "Year ended 31 .03 .2023", "31.12.2022"),
            ),
            {
                1: Column(date(2022, 12, 31), 6),
                2: Column(date(2023, 3, 31), 12),
                3: Column(date(2022, 12, 31), 9),
            },
            id="OCR's dotted length and spaced dates",
        ),
        pytest.param(
            (("Particulars", "31.03.2025", "31.03.2024"), ("Restated", "2024", "")),
            {1: Column(date(2025, 3, 31), None), 2: Column(date(2024, 3, 31), None)},
            id="a dated column keeps its date over a stray year",
        ),
    ],
)
def test_column_periods(header: tuple[tuple[str, ...], ...], expected: dict[int, Column]) -> None:
    assert column_periods(header) == expected


@pytest.mark.parametrize(
    ("text", "scale"),
    [
        ("(₹ in crore) ... (₹ in crore) ... in lac", 1e7),
        ("(₹ in lac) figures in lakhs", 1e5),
        ("USD in million", 1e6),
        pytest.param("the place where it was replaced", None, id="lac inside a word"),
        ("no scale here", None),
    ],
)
def test_document_scale(text: str, scale: float | None) -> None:
    assert document_scale(text) == scale
