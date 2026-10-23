"""The build's checks of what it loaded: the frozen fundamentals found and equal, and each filing's
own sums holding, which a misread period or scale would break."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from aigent.facts.cfg_facts import FILING_SUMS, FUNDAMENTALS_CONCEPTS
from aigent.facts.reference import Fundamental

Status = Literal["equal", "differs", "missing", "not in xbrl"]


@dataclass(frozen=True)
class Checked:
    key: str
    status: Status
    expected: float
    found: float | None


def tolerance(decimals: Iterable[int | None]) -> float:
    """Half a unit of the last filed digit, summed over the values compared; INF counts as 2 dp."""
    return sum(0.5 * 10.0 ** -(2 if d is None else d) for d in decimals)


def check_fundamentals(conn: sqlite3.Connection, table: Iterable[Fundamental]) -> list[Checked]:
    """Each fundamental against the consolidated year's figure in that year's own March filing."""
    out: list[Checked] = []
    for item in table:
        concept = FUNDAMENTALS_CONCEPTS.get(item.metric)
        if concept is None:
            out.append(Checked(item.key, "not in xbrl", item.value, None))
            continue
        row = conn.execute(
            """SELECT facts.value, facts.decimals FROM facts JOIN filings ON filings.id = filing_id
               WHERE facts.ticker = ? AND facts.basis = 'consolidated' AND facts.tier = 'xbrl'
                 AND concept = ? AND period_kind = 'duration' AND segment IS NULL
                 AND period_start = ? AND facts.period_end = ? AND filings.period_end = ?""",
            (
                item.ticker,
                concept,
                f"{item.year - 1}-04-01",
                f"{item.year}-03-31",
                f"{item.year}-03-31",
            ),
        ).fetchone()
        if row is None:
            out.append(Checked(item.key, "missing", item.value, None))
            continue
        value, decimals = float(row[0]), row[1]
        same = abs(value - item.value) <= tolerance([decimals])
        out.append(Checked(item.key, "equal" if same else "differs", item.value, value))
    return out


def check_sums(conn: sqlite3.Connection, filing_id: int) -> list[str]:
    """Every configured sum the filing breaks, for each period it files all the terms of."""
    terms: dict[tuple[str, str, str], dict[str, tuple[float, int | None]]] = {}
    for concept, kind, start, end, value, decimals in conn.execute(
        """SELECT concept, period_kind, period_start, period_end, value, decimals FROM facts
           WHERE filing_id = ? AND tier = 'xbrl' AND segment IS NULL""",
        (filing_id,),
    ):
        terms.setdefault((kind, start or "", end), {})[concept] = (float(value), decimals)
    failures: list[str] = []
    for (_, start, end), held in sorted(terms.items()):
        for total, needed, optional in FILING_SUMS:
            if total not in held or any(p not in held for p in needed):
                continue
            parts = (*needed, *(p for p in optional if p in held))
            added = sum(held[p][0] for p in parts)
            allowed = tolerance([held[total][1], *(held[p][1] for p in parts)])
            if abs(held[total][0] - added) > allowed:
                period = f"{start} to {end}" if start else end
                failures.append(
                    f"{total} {held[total][0]:.0f} != {' + '.join(parts)} {added:.0f} ({period})"
                )
    return failures
