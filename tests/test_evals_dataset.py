"""The loader's job is to fail before the run, not during it."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic.evals.dataset import DatasetError, digest, load_jsonl


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


def test_a_typo_in_a_field_name_is_refused(tmp_path: Path) -> None:
    # The whole point of extra="forbid": "expcted" would otherwise silently label nothing.
    path = _write(tmp_path, '{"id": "a1", "input": {}, "expcted": {"company": "Infosys"}}')
    with pytest.raises(DatasetError, match="expcted"):
        load_jsonl(path)


def test_duplicate_ids_are_refused_with_both_line_numbers(tmp_path: Path) -> None:
    path = _write(tmp_path, GOOD, GOOD)
    with pytest.raises(DatasetError, match="duplicate id 'a1'.*line 1"):
        load_jsonl(path)


def test_bad_json_names_its_line(tmp_path: Path) -> None:
    path = _write(tmp_path, GOOD, "{not json}")
    with pytest.raises(DatasetError, match="cases.jsonl:2"):
        load_jsonl(path)


def test_an_empty_dataset_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "", "// nothing here yet")
    with pytest.raises(DatasetError, match="no cases"):
        load_jsonl(path)


def test_digest_changes_when_a_label_changes(tmp_path: Path) -> None:
    before = digest(_write(tmp_path, GOOD))
    after = digest(_write(tmp_path, GOOD.replace("Infosys", "Wipro")))
    assert before != after, "a report's digest must move when the labels move"
    assert len(before) == 12
