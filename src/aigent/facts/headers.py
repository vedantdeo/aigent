"""A filed table's header read for what it states: each column's period (`column_periods`), and
the basis, scale and accounting framework the text around a table names. Dates, lengths and bases
are read through OCR's slips (`31 .12.2022`, `Six.months`, `Sectember`, `CONSOUDATED`)."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date

from aigent.facts.cfg_facts import (
    CROSSCHECK_FRAMEWORKS,
    CROSSCHECK_HEADING_LEAD,
    CROSSCHECK_NAVIGATION,
    CROSSCHECK_SCALES,
)
from aigent.facts.printed import NUMERIC_DATE, WORDED_DATE, YEAR
from aigent.facts.vocab import repair

MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        1,
    )
    for m in names
}
_DAY_FIRST_DATE = re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*([a-z]{3,9})\.?\s*,?\s*(\d{4})")
_APOSTROPHE_DATE = re.compile(  # 30 Jun'24, 30th Sep'25
    r"(\d[\do]?)\s*(?:st|nd|rd|th)?\s*([a-z]{3})[a-z]*\s*['’‘`]\s*(\d{2})\b"
)
_FY_SPAN = re.compile(r"(20\d{2})\s*[-–]\s*(\d{2})\b")  # 2024-25
_FY_SHORT = re.compile(r"\bfy\s*'?(\d{2}|20\d{2})\b")  # FY25, FY 2025
_SCALE = re.compile(r"\bin\s*(crore|lac|lakh|million)s?\b")
_TITLE_WORDS = 8  # a first-column header this long is the table's title, not a heading


@dataclass(frozen=True)
class Column:
    """A table column's period as its header states it: the day it ends, and its length in months,
    0 for a balance at that day, None when the header does not say."""

    end: date | None
    months: int | None


def month_of(word: str) -> int | None:
    """A month by its name, or for an OCR'd one by its first two letters when no other month
    starts so: `sen` is September, `jul` must be spelled out."""
    if word in MONTHS:
        return MONTHS[word]
    starting = {m for name, m in MONTHS.items() if name[:2] == word[:2]}
    return starting.pop() if len(starting) == 1 else None


def date_in(text: str) -> date | None:
    """The last whole date a text states, numeric (`31.03.2025`, OCR's `31 .03.2025`), worded
    either way round (`March 31, 2025`, `31st March, 2025`) or with an apostrophe (`30 Jun'24`)."""
    found: list[date] = []
    text = repair(text)
    closed = re.sub(r"(?<=\d)\s*([./-])\s*(?=\d)", r"\1", text)  # OCR's `31 .12.2022`
    for d, m, y in NUMERIC_DATE.findall(closed):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            continue
    for name, d, y in WORDED_DATE.findall(text.lower()):
        month = MONTHS.get(name)
        if month is None:
            continue
        try:
            found.append(date(int(y), month, int(d)))
        except ValueError:
            continue
    for d, name, y in _DAY_FIRST_DATE.findall(text.lower()):
        month = MONTHS.get(name)
        if month is None:
            continue
        try:
            found.append(date(int(y), month, int(d)))
        except ValueError:
            continue
    for d, name, y in _APOSTROPHE_DATE.findall(text.lower()):
        month = month_of(name)
        if month is None:
            continue
        try:
            found.append(date(2000 + int(y), month, int(d.replace("o", "0"))))
        except ValueError:
            continue
    return found[-1] if found else None


def fy_end_in(text: str) -> date | None:
    """The day a financial year stated as `2024-25`, `FY25` or `FY 2025` ends: 31 March 2025."""
    t = text.lower()
    for first, second in _FY_SPAN.findall(t):
        if (int(first) + 1) % 100 == int(second):
            return date(int(first) + 1, 3, 31)
    for year in _FY_SHORT.findall(t):
        return date(int(year) if len(year) == 4 else 2000 + int(year), 3, 31)
    return None


def months_in(text: str) -> int | None:
    """A header's length in months, 0 for a balance "as at" a day even at a half year's end."""
    t = repair(text).lower()
    if re.search(r"\bas\s*at\b|\basat\b|^\s*at\b", t):
        return 0
    if re.search(r"nine\W*months|\bni\S{1,3}\s+months|\b9m\b|\b9\s*month", t):
        return 9
    if re.search(r"six\W*months|\bsi\S{1,2}\s+months|half[ -]?year|\bh1\b|\b6\s*month", t):
        return 6
    if re.search(r"three\W*months|\b3\s*month|\b3m\b|quarter|\bq[1-4l]\b|\bq[1-4l]-", t):
        return 3
    if re.search(r"\byear\b|\bfy|twelve\W*months|\b12\s*month", t):
        return 12
    return None


def months_of(start: date | None, end: date) -> int:
    """A period's length in whole months, 0 for an instant."""
    if start is None:
        return 0
    return round(((end - start).days + 1) / 30.44)


def _completed(text: str, title: date | None) -> date | None:
    """A header's part of a date, its month and day or its year (OCR'd as `2024°`), completed
    from the title's date."""
    if title is None:
        return None
    t = repair(text).lower()
    years = re.findall(r"\b((?:19|20)\d{2})(?!\d)", t)
    month, day = title.month, title.day
    for name, d in re.findall(r"([a-z]{3,9})\.?\s*(\d{1,2})\b", t):
        if (found := month_of(name)) is not None:
            month, day = found, int(d)
    try:
        return date(int(years[-1]) if years else title.year, month, day)
    except ValueError:
        return None


def _partly_dated(text: str) -> bool:
    """Whether a header names a month, a year or an audit status, though no whole date."""
    found = re.findall(r"[a-z]+", repair(text).lower())
    return (
        any(w in MONTHS or w in ("audited", "unaudited") for w in found)
        or re.search(r"\b(?:19|20)\d{2}\b", text) is not None
    )


@dataclass
class _Said:
    """What a header says of each column: its own cells, a group heading carried over it from
    the left, the table's title above it, and a year alone among a row's other headers."""

    width: int
    own: dict[int, list[str]] = field(default_factory=dict[int, list[str]])
    carried: dict[int, list[str]] = field(default_factory=dict[int, list[str]])
    titled: dict[int, list[str]] = field(default_factory=dict[int, list[str]])
    years: dict[int, int] = field(default_factory=dict[int, int])

    def mine(self, j: int) -> str:
        return " ".join(self.own.get(j, []))

    def theirs(self, j: int) -> str:
        return " ".join(self.carried.get(j, []))

    def carried_months(self, j: int) -> int | None:
        """The length named by the heading carried over a column nearest its dates, the lowest."""
        return next(
            (m for h in reversed(self.carried.get(j, [])) if (m := months_in(h)) is not None), None
        )


def _said(header: Sequence[Sequence[str]]) -> _Said:
    """Each column's cells, the heading to its left carried over its empty cells (none from a
    line below the last line of dates, nor from a title), and the first column's title lent to
    the empty cells after it."""
    said = _Said(max((len(row) for row in header), default=0))
    dated = [
        i
        for i, row in enumerate(header)
        if any(date_in(c) or fy_end_in(c) or YEAR.fullmatch(c.strip()) for c in row)
    ]
    last_dated = max(dated, default=len(header) - 1)
    for i, row in enumerate(header):
        carry, title = "", ""
        filled = sum(bool(c.strip()) for c in row)
        for j in range(said.width):
            cell = row[j].strip() if j < len(row) else ""
            if cell:
                title = cell if j == 0 and len(cell.split()) >= _TITLE_WORDS else ""
                carry = cell if i <= last_dated and not title else ""
                said.own.setdefault(j, []).append(cell)
                if filled >= 2 and (year := YEAR.fullmatch(cell)):
                    said.years[j] = int(year.group(1))
            elif carry:
                said.carried.setdefault(j, []).append(carry)
            elif title:
                said.titled.setdefault(j, []).append(title)
    return said


def _dated(said: _Said, j: int) -> Column | None:
    """A column's period from what is said of it, the end tried in order: a date in its own
    cells, in the heading over it, or the two read together (`March 31,` over `2024`); the
    title's date completed by its own parts, for a column naming a length or part of a date; a
    financial year, 12 months unless said otherwise. Its length from its own cells, else the
    nearest heading over it that names one."""
    mine, theirs = said.mine(j), said.theirs(j)
    months = months_in(mine)
    months = said.carried_months(j) if months is None else months
    end = date_in(mine) or date_in(theirs) or date_in(f"{theirs} {mine}")
    if end is None and (_partly_dated(mine) or months_in(mine) is not None):
        end = _completed(mine, date_in(" ".join(said.titled.get(j, []))))
    if end is None and (end := fy_end_in(mine) or fy_end_in(theirs)) is not None:
        months = 12 if months is None else months
    return None if end is None else Column(end, months)


def _shared_day(said: _Said, columns: dict[int, Column]) -> dict[int, Column]:
    """A year alone among a row's other headers (`2025` beside `31 March 2024`) dated by the day
    the dated columns share, when they share one."""
    days = {(c.end.month, c.end.day) for c in columns.values() if c.end is not None}
    if len(days) != 1:
        return columns
    month, day = days.pop()
    out = dict(columns)
    for j, year in said.years.items():
        if j not in out:
            months = months_in(said.mine(j))
            months = said.carried_months(j) if months is None else months
            out[j] = Column(date(year, month, day), months)
    return out


def _neighbours_length(said: _Said, columns: dict[int, Column]) -> dict[int, Column]:
    """A dated column still of no length takes its left neighbour's, else the length its right
    neighbour's header names ("Quarter ended" over the middle of three), left to right."""
    named = {j: m for j in range(1, said.width) if (m := months_in(said.mine(j))) is not None}
    out = dict(columns)
    for j in sorted(out):
        if out[j].months is not None:
            continue
        left = out[j - 1].months if j - 1 in out else named.get(j - 1)
        months = left if left is not None else named.get(j + 1)
        if months is not None:
            out[j] = replace(out[j], months=months)
    return out


_RESOLVE: tuple[Callable[[_Said, dict[int, Column]], dict[int, Column]], ...] = (
    _shared_day,
    _neighbours_length,
)  # in order: the order is the rule


def column_periods(header: Sequence[Sequence[str]]) -> dict[int, Column]:
    """Each column's period from the header rows: what is said of each column (`_said`), each
    column dated from it (`_dated`), then the table-wide steps in `_RESOLVE`. The first column,
    the labels', takes no period."""
    said = _said(header)
    columns = {j: c for j in range(1, said.width) if (c := _dated(said, j)) is not None}
    for step in _RESOLVE:
        columns = step(said, columns)
    return columns


def scale_in(text: str) -> float | None:
    """The last scale the text states as "in crore", "in lac" and the like."""
    found = _SCALE.findall(text.lower())
    return CROSSCHECK_SCALES[found[-1]] if found else None


def document_scale(text: str) -> float | None:
    """The scale a document states most often as "in crore", "in lac" and the like."""
    found = _SCALE.findall(text.lower())
    if not found:
        return None
    return CROSSCHECK_SCALES[max(set(found), key=found.count)]


def basis_in(text: str) -> str | None:
    """The last basis the text names, "standalone" or "consolidated", read through OCR's slips
    (`repair`: `CONSOUDATED`)."""
    found = re.findall(r"standalone|consolidated", repair(text).lower())
    return found[-1] if found else None


def section_basis(text: str) -> str | None:
    """The basis the text's last heading names, starting a section: a line naming one basis,
    capitalised (`Standalone`, `CONSOLIDATED`) or in its first words (`Statement of standalone
    results`); prose, a line naming both and a report's navigation strip name none."""
    found = None
    for line in text.splitlines():
        if re.search(CROSSCHECK_NAVIGATION, line, re.IGNORECASE):
            continue
        words = re.findall(r"[A-Za-z]+", line)
        named = {basis_in(w): (i, w) for i, w in enumerate(words) if basis_in(w)}
        if len(named) == 1 and all(
            w[0].isupper() or i < CROSSCHECK_HEADING_LEAD for i, w in named.values()
        ):
            found = next(iter(named))
    return found


def framework_in(text: str) -> str | None:
    """The last accounting framework the text names (`CROSSCHECK_FRAMEWORKS`): "ind as" or
    "ifrs"; a standard cited by number (`IFRS 16`) names none."""
    lowered = text.lower()
    found = [
        (m.start(), name)
        for name, pattern in CROSSCHECK_FRAMEWORKS.items()
        for m in re.finditer(pattern, lowered)
    ]
    return max(found)[1] if found else None
