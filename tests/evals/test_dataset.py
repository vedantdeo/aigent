"""The loader's job is to fail before the run, not during it."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic.errors import DatasetError
from entropic.evals.dataset import digest, load_jsonl


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


GOOD = (
    '{"id": "a1", "input": {"headline": "Infosys Q3 profit up 12%"}, '
    '"expected": {"company": "Infosys"}}'
)


def test_loads_a_well_formed_dataset(tmp_path: Path) -> None:
    path = _write(tmp_path, GOOD, '{"id": "a2", "input": {"headline": "TCS flat"}, "tags": ["q3"]}')
    cases = load_jsonl(path)

    assert [case.id for case in cases] == ["a1", "a2"]
    assert cases[0].expected == {"company": "Infosys"}
    assert cases[1].expected == {}, "expected defaults to empty, not None"
    assert cases[1].tags == ["q3"]


def test_blank_and_commented_lines_are_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path, GOOD, "", "// still arguing with myself about this one", "   ")
    assert len(load_jsonl(path)) == 1


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        pytest.param(
            ('{"id": "a1", "input": {}, "expcted": {"company": "Infosys"}}',),
            "expcted",
            # The whole point of extra="forbid": "expcted" would otherwise silently label nothing.
            id="a typo in a field name",
        ),
        pytest.param(
            (GOOD, GOOD), r"duplicate id 'a1'.*line 1", id="a duplicate id, with both lines"
        ),
        pytest.param((GOOD, "{not json}"), "cases.jsonl:2", id="bad JSON, naming its line"),
        pytest.param(("", "// nothing here yet"), "no cases", id="a file with nothing in it"),
    ],
)
def test_the_loader_refuses(tmp_path: Path, lines: tuple[str, ...], match: str) -> None:
    with pytest.raises(DatasetError, match=match):
        load_jsonl(_write(tmp_path, *lines))


def test_digest_changes_when_a_label_changes(tmp_path: Path) -> None:
    before = digest(_write(tmp_path, GOOD))
    after = digest(_write(tmp_path, GOOD.replace("Infosys", "Wipro")))
    assert before != after, "a report's digest must move when the labels move"
    assert len(before) == 12
