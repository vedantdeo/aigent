"""The frozen fundamentals: 24 figures, every value cited to a page of its own year's filing."""

from __future__ import annotations

import json
import re
from typing import cast

import pytest

from aigent.facts.reference import FUNDAMENTALS, Fundamental, load_fundamentals
from aigent.retrieval.corpus import CORPUS_DIR, load_pdf_cached

TABLE = load_fundamentals()
FIELDS = ["value", "unit", "printed", "doc_id", "page", "note"]


def test_the_table_holds_four_metrics_for_two_years_of_three_companies() -> None:
    assert len(TABLE) == 24
    assert {(f.ticker, f.year) for f in TABLE.values()} == {
        (t, y) for t in ("ITC", "RELIANCE", "TATAMOTORS") for y in (2024, 2025)
    }


def test_the_file_is_sorted_at_every_level_with_one_shape_per_entry() -> None:
    raw = cast(
        dict[str, dict[str, dict[str, dict[str, object]]]],
        json.loads(FUNDAMENTALS.read_text())["fundamentals"],
    )
    assert list(raw) == sorted(raw)
    for years in raw.values():
        assert list(years) == sorted(years)
        for metrics in years.values():
            assert list(metrics) == sorted(metrics)
            assert all(list(entry) == FIELDS for entry in metrics.values())


def test_each_value_is_cited_to_its_own_year_s_report() -> None:
    for f in TABLE.values():
        assert f.doc_id == f"{f.ticker}-FY{f.year % 100}", f.key


def _page(f: Fundamental) -> str:
    document = load_pdf_cached(CORPUS_DIR / f"{f.doc_id}.pdf")
    starts = (*document.page_starts, len(document.text))
    return document.text[starts[f.page - 1] : starts[f.page]].replace(",", "")


@pytest.mark.skipif(
    not all((CORPUS_DIR / f"{f.doc_id}.pdf").exists() for f in TABLE.values()),
    reason="the filings are not fetched (./scripts/fetch-corpus.sh)",
)
@pytest.mark.parametrize("fundamental", list(TABLE.values()), ids=lambda f: f.key)
def test_every_printed_number_is_on_the_cited_page(fundamental: Fundamental) -> None:
    page = _page(fundamental)
    numbers = [n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", fundamental.printed)]
    missing = [n for n in numbers if n not in page and n.rstrip("0").rstrip(".") not in page]
    assert not missing, f"{fundamental.key}: {missing} not on {fundamental.citation}"
