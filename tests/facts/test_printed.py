"""`facts.printed`: a printed figure read to its value, and where a table's body begins."""

from __future__ import annotations

import pytest

from aigent.facts.printed import figure_row, number


@pytest.mark.parametrize(
    ("printed", "read"),
    [
        ("38,995.78", (38995.78, 2, False)),
        ("1,07,000", (107000.0, 0, False)),
        ("59923,95", (59923.95, 2, False)),
        ("1.407.45", (1407.45, 2, False)),
        ("2652. 72", (2652.72, 2, False)),
        (".. 14,692.66", (14692.66, 2, False)),
        ("(1,937.10)", (-1937.10, 2, False)),
        ("(29642.01", (-29642.01, 2, False)),
        ("-12.5", (-12.5, 1, False)),
        ("2.15%", (2.15, 2, True)),
        ("..", (0.0, 0, False)),
        ("-", (0.0, 0, False)),
        ("—", (0.0, 0, False)),
        pytest.param("(1;331)", (-1331.0, 0, False), id="OCR's semicolon for a separator"),
        pytest.param("5,5,959.31", (5959.31, 2, False), id="OCR's doubled lead group"),
        pytest.param("5,59,593", (559593.0, 0, False), id="an Indian lakh group is not doubled"),
        pytest.param("[555)", (-555.0, 0, False), id="OCR's square bracket opening a negative"),
        pytest.param("15.73 15.46", None, id="two numbers in a cell"),
        pytest.param("31.03.2025", None, id="a date"),
        pytest.param("June 30, 2024", None, id="a worded date"),
        pytest.param("Interest earned", None, id="text"),
        pytest.param("", None, id="an empty cell"),
    ],
)
def test_number(printed: str, read: tuple[float, int, bool] | None) -> None:
    assert number(printed) == read


@pytest.mark.parametrize(
    ("row", "figures"),
    [
        pytest.param(("Revenue", "100", "90"), True, id="a line of figures"),
        pytest.param(("Revenue", "100", ""), False, id="one figure alone"),
        pytest.param(("", "2025", "2025*", "2024*"), False, id="a header's years, some restated"),
        pytest.param(("Particulars", "2025", "2024"), False, id="a header's bare years"),
        pytest.param(("Employees", "2025", "1,840"), True, id="a year beside another figure"),
    ],
)
def test_figure_row(row: tuple[str, ...], figures: bool) -> None:
    assert figure_row(row) is figures
