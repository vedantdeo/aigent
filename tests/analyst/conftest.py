"""A toolbox over scripted specialists, shared by the analyst's tests."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from aigent.analyst.experts import figures_expert
from aigent.analyst.router import Router
from aigent.analyst.specialists import FigureJob, Passage, Specialist, Told, check_figures
from aigent.analyst.toolbox import Toolbox
from aigent.facts.store import connect
from aigent.retrieval.chunk import context_block

from ..conftest import PASSAGES

DIVIDEND = "RELIANCE-FY25#0001"  # "The Board recommended a dividend of Rs 5.50 per equity share."
READ = Passage(DIVIDEND, PASSAGES["RELIANCE-FY25"][0].citation, PASSAGES["RELIANCE-FY25"][0].text)
FIGURE = (
    '{"metric": "dividend", "value": 5.5, "unit": "INR", "year": 2025, "description": null, '
    f'"chunk_id": "{DIVIDEND}"}}'
)


def told(text: str, *read: Passage) -> Told:
    """A reply that read `read`, the dividend passage when none is named."""
    return Told(text, 0.01, 100, 10, (), read or (READ,))


def never(job: object) -> Told:
    raise AssertionError("the generalist was asked")


def box(
    served: Callable[[str, dict[str, object]], tuple[str, bool]],
    figures: Callable[[FigureJob], Told] = lambda job: told('{"figures": [' + FIGURE + "]}"),
) -> Toolbox:
    return Toolbox(
        served_tools=[{"name": "search_filings", "input_schema": {"type": "object"}}],
        served=served,
        route=Router(price=lambda i, o: 0.0),
        experts=[
            figures_expert(Specialist("extract_figures", figures, never, check_figures)),
        ],
    )


def search(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
    assert name == "search_filings"
    return context_block([c for chunks in PASSAGES.values() for c in chunks]), False


def _fact(
    ticker: str,
    year: int,
    value: float,
    tier: str = "xbrl-filed",
    concept: str = "OperatingProfit",
    unit: str = "INR",
    instant: bool = False,
) -> dict[str, object]:
    period = (
        {"as_at": f"{year}-03-31"}
        if instant
        else {"from": f"{year - 1}-04-01", "to": f"{year}-03-31"}
    )
    return {
        "cite_as": f"FACT:{ticker}:{year}:standalone:year:{concept}",
        "label": concept,
        "value": value,
        "unit": unit,
        "period": period,
        "basis": "standalone",
        "segment": None,
        "tier": tier,
        "source": "NSE XBRL results filing",
    }


OP25 = _fact("ICICIBANK", 2025, 672988200000.0)
OP24 = _fact("ICICIBANK", 2024, 581307600000.0)
HDFC25 = _fact("HDFCBANK", 2025, 1000000000000.0)
UNVERIFIED = _fact("ICICIBANK", 2026, 1.0, tier="xbrl-ai")
ADV25 = _fact("ICICIBANK", 2025, 800.0, concept="Advances", instant=True)
DEP25 = _fact("ICICIBANK", 2025, 1000.0, concept="Deposits", instant=True)
EPS25 = _fact("ICICIBANK", 2025, 67.01, concept="Eps", unit="INRPerShare")
HELD = (OP25, OP24, HDFC25, ADV25, DEP25, EPS25)


def evidence_of(*facts: dict[str, object]) -> dict[str, Passage]:
    """Facts as `fundamentals` puts them into a toolbox's evidence, by cite_as."""
    return {str(f["cite_as"]): Passage(str(f["cite_as"]), "s", json.dumps(f)) for f in facts}


def _metric(
    conn: sqlite3.Connection, name: str, formula: str, bindings: dict[str, tuple[str, str]]
) -> None:
    conn.execute(
        """INSERT INTO derived_concepts (name, scope, label, formula, bindings, tier, written_by,
                  written_at, run_id)
           VALUES (?, 'banking', ?, ?, ?, 'derived-concept-ai', 'seed', 't', 'seed-1')""",
        (
            name,
            f"{name} in words",
            formula,
            json.dumps({v: {"concept": c, "offset": o} for v, (c, o) in sorted(bindings.items())}),
        ),
    )


@pytest.fixture
def concepts_db(tmp_path: Path) -> sqlite3.Connection:
    """A facts database holding the verified facts above, their concepts, and three metrics:
    OperatingProfitGrowth, CreditDepositRatio and OperatingProfitLessEps (units that differ)."""
    conn = connect(tmp_path / "facts.sqlite")
    with conn:
        for fact in HELD:
            _, ticker, _, _, _, concept = str(fact["cite_as"]).split(":")
            period = cast(dict[str, str], fact["period"])
            conn.execute(
                """INSERT INTO facts (ticker, basis, concept, label, period_kind, period_start,
                          period_end, value, unit, tier, written_by, written_at, run_id,
                          verified_by, verified_at)
                   VALUES (?, 'standalone', ?, ?, ?, ?, ?, ?, ?, 'xbrl-filed', 'w', 't', 'r',
                           'nse-pdf:x', 't')""",
                (
                    ticker,
                    concept,
                    concept,
                    "instant" if "as_at" in period else "duration",
                    period.get("from"),
                    period.get("to", period.get("as_at")),
                    fact["value"],
                    fact["unit"],
                ),
            )
            conn.execute(
                "INSERT OR IGNORE INTO concepts (concept, label, kind) VALUES (?, ?, 'xbrl')",
                (concept, concept),
            )
        _metric(
            conn,
            "OperatingProfitGrowth",
            "growth",
            {"current": ("OperatingProfit", "t"), "previous": ("OperatingProfit", "t-1y")},
        )
        _metric(
            conn,
            "CreditDepositRatio",
            "ratio",
            {"numerator": ("Advances", "t"), "denominator": ("Deposits", "t")},
        )
        _metric(
            conn,
            "OperatingProfitLessEps",
            "difference",
            {"minuend": ("OperatingProfit", "t"), "subtrahend": ("Eps", "t")},
        )
    return conn
