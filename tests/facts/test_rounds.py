"""`facts.rounds`: two runs compared fact by fact, tables ranked by facts that differ, and the
audit command's totals."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from aigent.facts import __main__ as cli
from aigent.facts.documents import Entry
from aigent.facts.rounds import RankedTable, diff_report, diff_runs, rank_tables, read_rows


def _row(fact: int, outcome: str, value: float = 5.0, **more: object) -> dict[str, object]:
    return {"fact_id": fact, "outcome": outcome, "value": value, **more}


def test_read_rows_keys_a_run_by_fact(tmp_path: Path) -> None:
    path = tmp_path / "run.rows.jsonl"
    path.write_text(json.dumps(_row(7, "agree")) + "\n\n" + json.dumps(_row(9, "differ")) + "\n")
    assert sorted(read_rows(path)) == [7, 9]


_BEFORE = {
    1: _row(1, "agree", document="a.pdf", sheet=2),
    2: _row(2, "agree", value=0.0, document="a.pdf", sheet=2),
    3: _row(3, "unmatched", document="b.pdf", sheet=None),
    4: _row(4, "differ", document="b.pdf", sheet=1),
    5: _row(5, "agree", printed="1.0"),
}
_AFTER = {
    1: _row(1, "differ", document="a.pdf", sheet=2, veto="row turns: Profit"),
    2: _row(2, "unmatched", value=0.0, document="a.pdf", sheet=2, veto=None),
    3: _row(3, "agree", document="b.pdf", sheet=0, veto=None),
    4: _row(4, "differ", document="b.pdf", sheet=1, veto=None),
    5: _row(5, "agree", printed="1.00", veto=None),
    6: _row(6, "agree"),
}


def test_diff_runs_counts_what_moved() -> None:
    diff = diff_runs(_BEFORE, _AFTER)
    assert (diff.lost, diff.won, diff.lost_zero, diff.won_zero) == ((1, 2), (3,), 1, 0)
    assert diff.outcomes[0]["agree"] == 3 and diff.outcomes[1]["agree"] == 2
    assert diff.fields == {"outcome": 3, "sheet": 1, "printed": 1}, diff.fields


def test_a_veto_is_compared_only_where_both_runs_have_it() -> None:
    before = {1: _row(1, "unmatched", veto="opposites: x")}
    assert diff_runs(before, {1: _row(1, "unmatched", veto=None)}).fields == {"veto": 1}
    assert diff_runs({1: _row(1, "unmatched")}, before).fields == {}


def test_the_report_names_tables_and_vetoes() -> None:
    report = diff_report(diff_runs(_BEFORE, _AFTER), _BEFORE, _AFTER)
    assert "lost 2 (1 zero-valued), won 1 (0 zero-valued)" in report
    assert "  2 a.pdf #2" in report and "  1 b.pdf #0" in report
    assert "  1 row turns" in report, report


@pytest.mark.parametrize(
    ("readings", "ranked"),
    [
        pytest.param(
            True,
            [RankedTable("r.pdf#rapidocr", 1, 3, 0), RankedTable("a.pdf", 2, 2, 1)],
            id="most differing first",
        ),
        pytest.param(False, [RankedTable("a.pdf", 2, 2, 1)], id="readings left out"),
    ],
)
def test_rank_tables(readings: bool, ranked: list[RankedTable]) -> None:
    rows = [
        *(_row(n, "differ", document="r.pdf#rapidocr", sheet=1) for n in range(3)),
        _row(10, "differ", document="a.pdf", sheet=2),
        _row(11, "differ", document="a.pdf", sheet=2),
        _row(12, "agree", document="a.pdf", sheet=2),
        _row(13, "unmatched", document="a.pdf", sheet=None),
    ]
    assert rank_tables(rows, 5, readings) == ranked


def test_rank_tables_keeps_the_top_only() -> None:
    rows = [_row(n, "differ", document=f"{n}.pdf", sheet=0) for n in range(4)]
    assert [t.document for t in rank_tables(rows, 2)] == ["0.pdf", "1.pdf"]


def test_table_audit_totals_flags_by_kind_and_writes_every_table(tmp_path: Path) -> None:
    manifest: dict[str, Entry] = {
        "results/A-2025-03-31.pdf": {
            "format": "pdf",
            "source": "nse",
            "pages": 2,
            "text_less_pages": [],
            "ocr_pages": [],
        },
    }

    def audit(readings: object) -> dict[str, dict[str, object]]:
        flags = {"merged figures": 1, "no period read": 0}
        return {
            "results/A-2025-03-31.pdf#0": {"rows": [["x"]], "cells": 4, "flags": flags},
            "results/A-2025-03-31.pdf#3": {
                "rows": [],
                "cells": 2,
                "flags": dict.fromkeys(flags, 0),
            },
        }

    out = cli.table_audit(
        datetime(2026, 11, 25, 22, 0), tmp_path, audit, manifest, ["results/A-2025-03-31.pdf"]
    )
    assert "tables 2, cells 6, flagged 1, merged figures 1, no period read 0" in out, out
    written = json.loads((tmp_path / "table-audit-20261125-2200.json").read_text())
    assert sorted(written) == ["results/A-2025-03-31.pdf#0", "results/A-2025-03-31.pdf#3"]
