"""The frozen fundamentals: each company's headline figures per year, every value cited to the
page of the filing it came from. The facts database's build checks itself against them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

FUNDAMENTALS = Path(__file__).resolve().parents[3] / "evals" / "reference" / "fundamentals.json"


@dataclass(frozen=True)
class Fundamental:
    """One company's figure for one year, with the filing page it is printed on."""

    ticker: str
    year: int
    metric: str
    value: float
    unit: str
    printed: str
    doc_id: str
    page: int
    note: str | None

    @property
    def key(self) -> str:
        """How a claim cites it: ticker, year and metric."""
        return f"{self.ticker}:{self.year}:{self.metric}"

    @property
    def citation(self) -> str:
        return f"{self.doc_id} p{self.page}"


Table = dict[tuple[str, int, str], Fundamental]


def load_fundamentals(path: Path = FUNDAMENTALS) -> Table:
    """The table, keyed by (ticker, year, metric)."""
    raw = cast(
        dict[str, dict[str, dict[str, dict[str, object]]]],
        json.loads(path.read_text(encoding="utf-8"))["fundamentals"],
    )
    table: Table = {}
    for ticker, years in raw.items():
        for year, metrics in years.items():
            for metric, entry in metrics.items():
                note = entry["note"]
                table[(ticker, int(year), metric)] = Fundamental(
                    ticker=ticker,
                    year=int(year),
                    metric=metric,
                    value=float(cast(float, entry["value"])),
                    unit=str(entry["unit"]),
                    printed=str(entry["printed"]),
                    doc_id=str(entry["doc_id"]),
                    page=int(cast(int, entry["page"])),
                    note=None if note is None else str(note),
                )
    return table
