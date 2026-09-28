"""What holds for all five patterns at once."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from aigent.workflows import (
    chaining,
    evaluator_optimizer,
    orchestrator_workers,
    parallelization,
    routing,
)
from aigent.workflows.reports import REPORTS, DocId

PATTERNS = (chaining, routing, parallelization, orchestrator_workers, evaluator_optimizer)


def test_the_report_type_names_exactly_the_reports_in_the_manifest() -> None:
    """`DocId` is what holds a model's choice of report to a real one, and it is written out by
    hand; the manifest is what the corpus is fetched from."""
    assert list(get_args(DocId)) == list(REPORTS)


@pytest.mark.parametrize("pattern", PATTERNS, ids=lambda module: module.__name__.split(".")[-1])
def test_each_pattern_fits_in_under_a_hundred_lines(pattern: object) -> None:
    """The roadmap's constraint, and the point of it: the pattern is the small part."""
    lines = Path(str(getattr(pattern, "__file__", ""))).read_text(encoding="utf-8").splitlines()
    assert len(lines) < 100, f"{len(lines)} lines"
