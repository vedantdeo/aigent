"""A printed figure as a filed table prints it: Indian grouping, brackets, dashes and stray OCR
separators read to a value. Free of the cross-check, so the documents it reads can use it too."""

from __future__ import annotations

import re
from collections.abc import Sequence

ZERO = {"..", "-", "–", "—", "nil", "0"}
NUMERIC_DATE = re.compile(r"(\d{1,2})[.,/-](\d{1,2})[.,/-](\d{4})")
WORDED_DATE = re.compile(r"([a-z]{3,9})\.?\s*(\d{1,2})\s*,?\s*(\d{4})")
YEAR = re.compile(r"((?:19|20)\d{2})\s*\*?")  # a header's year alone, as `2025` or `2024*`
MAIN_LINE = re.compile(r"\s*(?:[IVX]{1,5}|[A-H])[.)]?\s")  # IX, XI, D., II. open a main line
ENUMERATOR = re.compile(r"\(?(?:[a-z]|[ivx]+|\d{1,2})[.)](?=\s|$)")  # a), (ii), 9. open an item
FIGURE = re.compile(r"(?!\d+\.$)[(\[{]?[-+]?[\d,.]*\d[\d,.]*%?[)\]}]?%?")  # (12) {99) 9.8%; no `1.`
NIL = r"\.\.|[-–—]"  # what prints a nil figure


def number(text: str) -> tuple[float, int, bool] | None:
    """A cell as a printed number: its value, decimal places and whether it is a percentage; None
    for text, a date, or two numbers in one cell. Dashes and `..` print a zero."""
    t = text.strip().replace("−", "-")
    if t.lower() in ZERO:
        return 0.0, 0, False
    if NUMERIC_DATE.search(t) or WORDED_DATE.search(t.lower()):
        return None
    t = re.sub(r"^[.\s~]+(?=[\d(])", "", t)  # a leader of dots before the figure
    t = re.sub(r"^[\[{](?=\d)", "(", re.sub(r"(?<=\d)\}$", ")", t))  # OCR's `[555)`, `{99}`
    t = re.sub(r"(?<=\d);(?=\d)", ",", t)  # OCR's `1;331`
    if re.fullmatch(r"[\d,]+\.\s+\d+", t):  # a stray space after the point
        t = re.sub(r"\s+", "", t)
    match = re.fullmatch(r"(\()?(-)?([\d,.]+)(\))?\s*(%)?", t)
    if not match or not re.search(r"\d", match.group(3)):
        return None
    digits = match.group(3)
    if re.match(r"(\d),\1,\d", digits):  # OCR's doubled lead, 5,5,947: no group is one digit
        digits = digits[2:]
    if digits.count(".") == 0 and re.search(r",\d{2}$", digits):  # a decimal comma
        digits = digits[: digits.rfind(",")] + "." + digits[digits.rfind(",") + 1 :]
    if digits.count(".") > 1:  # points misread for separators
        head, _, tail = digits.rpartition(".")
        digits = head.replace(".", "") + "." + tail
    digits = digits.replace(",", "")
    if not re.fullmatch(r"\d+(\.\d+)?", digits):
        return None
    places = len(digits.partition(".")[2])
    value = float(digits)
    if match.group(1) or match.group(2):  # a closing bracket can fall in the next cell
        value = -value
    return value, places, bool(match.group(5))


def figure_row(row: Sequence[str]) -> bool:
    """A row printing two or more figures, not only years as a header's `2025 | 2024*` does."""
    figures = [c for c in row if number(c) is not None]
    return len(figures) >= 2 and not all(YEAR.fullmatch(c.strip()) for c in figures)


def is_figure(text: str) -> bool:
    """A word printing one figure, as `1,234.5`, `(12)`, OCR's `[555)` or `(9.8%)`."""
    return bool(FIGURE.fullmatch(text.strip()))


def figures_only(text: str) -> bool:
    """A cell printing figures, or a dash for none, only."""
    found = text.split()
    return bool(found) and all(is_figure(t) or number(t) is not None for t in found)
