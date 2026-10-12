"""The analyst's `fundamentals` tool: a frozen table of each company's headline figures per year,
every value cited to the page of the filing it came from, so it can be cited like a passage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from anthropic.types import ToolParam

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


def lookup(table: Table, ticker: str, metric: str, year: int) -> Fundamental:
    """The figure, or ValueError naming what the table does not hold."""
    found = table.get((ticker, year, metric))
    if found is not None:
        return found
    tickers = sorted({t for t, _, _ in table})
    if ticker not in tickers:
        raise ValueError(f"no company {ticker!r}; one of {', '.join(tickers)}")
    metrics = sorted({m for _, _, m in table})
    if metric not in metrics:
        raise ValueError(f"no metric {metric!r}; one of {', '.join(metrics)}")
    years = sorted({y for t, y, _ in table if t == ticker})
    raise ValueError(f"no {year} figures for {ticker}; years held: {', '.join(map(str, years))}")


_TABLE = load_fundamentals()

FUNDAMENTALS_TOOL: ToolParam = {
    "name": "fundamentals",
    "description": (
        "Look up a company's headline figure for a financial year from a frozen table: revenue "
        "from operations, profit attributable to owners, basic EPS, or the year's dividend per "
        "share, all consolidated and as originally reported. Each answer names the filing page "
        "it came from; cite it by `cite_as`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "enum": sorted({t for t, _, _ in _TABLE})},
            "metric": {"type": "string", "enum": sorted({m for _, _, m in _TABLE})},
            "year": {
                "type": "string",
                "enum": [str(y) for y in sorted({y for _, y, _ in _TABLE})],
                "description": "The financial year a report ends: 2025 is FY 2024-25.",
            },
        },
        "required": ["ticker", "metric", "year"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class FundamentalsTool:
    """`fundamentals` over one table, recording every lookup; an unknown one comes back as an
    error message, never an exception."""

    table: Table
    looked_up: list[str] = field(default_factory=list[str])

    def __call__(self, arguments: dict[str, object]) -> str:
        ticker, metric = str(arguments.get("ticker", "")), str(arguments.get("metric", ""))
        year = str(arguments.get("year", ""))
        try:
            found = lookup(self.table, ticker, metric, int(year) if year.isdigit() else -1)
        except ValueError as error:
            return f"Error: {error}"
        self.looked_up.append(found.key)
        return json.dumps(
            {
                "cite_as": found.key,
                "value": found.value,
                "unit": found.unit,
                "printed": found.printed,
                "source": found.citation,
                "note": found.note,
            }
        )
