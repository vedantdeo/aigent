"""A printed figure as a filed table prints it: Indian grouping, brackets, dashes and stray OCR
separators read to a value. Free of the cross-check, so the documents it reads can use it too."""

from __future__ import annotations

import re

ZERO = {"..", "-", "–", "—", "nil", "0"}
NUMERIC_DATE = re.compile(r"(\d{1,2})[.,/-](\d{1,2})[.,/-](\d{4})")
WORDED_DATE = re.compile(r"([a-z]{3,9})\.?\s*(\d{1,2})\s*,?\s*(\d{4})")


def number(text: str) -> tuple[float, int, bool] | None:
    """A cell as a printed number: its value, decimal places and whether it is a percentage; None
    for text, a date, or two numbers in one cell. Dashes and `..` print a zero."""
    t = text.strip().replace("−", "-")
    if t.lower() in ZERO:
        return 0.0, 0, False
    if NUMERIC_DATE.search(t) or WORDED_DATE.search(t.lower()):
        return None
    t = re.sub(r"^[.\s~]+(?=[\d(])", "", t)  # a leader of dots before the figure
    if re.fullmatch(r"[\d,]+\.\s+\d+", t):  # a stray space after the point
        t = re.sub(r"\s+", "", t)
    match = re.fullmatch(r"(\()?(-)?([\d,.]+)(\))?\s*(%)?", t)
    if not match or not re.search(r"\d", match.group(3)):
        return None
    digits = match.group(3)
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
