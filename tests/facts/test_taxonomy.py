"""The taxonomy packages: fetched once, read from the zip, merged into one label per concept."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from aigent.facts.taxonomy import (
    Labels,
    fetch_packages,
    labels_in,
    load_labels,
    merged,
    read_manifest,
    write_manifest,
)

from .conftest import FakeNse, label_package

NOW = datetime(2026, 10, 23, 10, 0, 0)
A, B = "https://x/a.zip", "https://x/b.zip"


def test_labels_are_read_by_concept_and_role_from_every_label_file() -> None:
    package = label_package(
        {
            "RevenueFromOperations": [("label", "Revenue from operations")],
            "Assets": [("label", "Assets"), ("totalLabel", "Total  assets")],
        }
    )
    assert labels_in(package) == {
        "Assets": {"label": {"Assets"}, "totalLabel": {"Total assets"}},
        "RevenueFromOperations": {"label": {"Revenue from operations"}},
    }


def test_a_prefix_with_hyphens_leaves_the_concept_whole() -> None:
    package = label_package({"Borrowings_Current": [("label", "B")]})
    assert list(labels_in(package)) == ["Borrowings_Current"]


@pytest.mark.parametrize(
    ("packages", "expected"),
    [
        pytest.param(
            [{"X": {"label": {"X first"}}}, {"X": {"label": {"X second"}}}],
            Labels("X first", {"X second"}),
            id="the first package's label wins, the next's is an alias",
        ),
        pytest.param(
            [{"X": {"label": {"B second", "A first"}}}],
            Labels("A first", {"B second"}),
            id="two standard labels in one package: the first in order wins",
        ),
        pytest.param(
            [{"X": {"label": {"Same"}}}, {"X": {"label": {"Same"}}}],
            Labels("Same", set()),
            id="the same label twice is no alias",
        ),
        pytest.param(
            [{"X": {"label": {"X"}, "totalLabel": {"Total X"}, "terseLabel": {"X"}}}],
            Labels("X", {"Total X"}),
            id="other roles are aliases",
        ),
        pytest.param(
            [{"X": {"totalLabel": {"Total X"}}}, {"X": {"label": {"X"}}}],
            Labels("X", set()),
            id="a package with no standard label gives none",
        ),
    ],
)
def test_packages_merge_into_one_label_per_concept(
    packages: list[dict[str, dict[str, set[str]]]], expected: Labels
) -> None:
    assert merged(packages)["X"] == expected


def test_packages_are_fetched_once_and_kept_with_their_digest(tmp_path: Path) -> None:
    nse = FakeNse({A: b"a", B: b"b"})
    assert fetch_packages(nse, tmp_path, [("one", A), ("two", B)], NOW) == ["one", "two"]
    assert fetch_packages(nse, tmp_path, [("one", A), ("two", B)], NOW) == []
    assert nse.calls == [A, B]
    manifest = read_manifest(tmp_path)
    assert list(manifest) == ["one", "two"]
    assert manifest["one"] == {
        "fetched": "2026-10-23T10:00:00",
        "sha256": hashlib.sha256(b"a").hexdigest(),
        "url": A,
    }


def test_a_new_url_is_fetched_again(tmp_path: Path) -> None:
    nse = FakeNse({A: b"a", B: b"b"})
    fetch_packages(nse, tmp_path, [("one", A)], NOW)
    assert fetch_packages(nse, tmp_path, [("one", B)], NOW) == ["one"]
    assert (tmp_path / "one.zip").read_bytes() == b"b"


def test_the_manifest_is_written_sorted(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"b": {"url": "u", "fetched": "f"}, "a": {"url": "v", "sha256": "s"}})
    written = json.loads((tmp_path / "manifest.json").read_text())
    assert [(name, list(entry)) for name, entry in written.items()] == [
        ("a", ["sha256", "url"]),
        ("b", ["fetched", "url"]),
    ]


def test_cached_packages_load_in_priority_order(tmp_path: Path) -> None:
    nse = FakeNse(
        {A: label_package({"X": [("label", "From A")]}), B: label_package({"X": [("label", "B")]})}
    )
    fetch_packages(nse, tmp_path, [("a", A), ("b", B)], NOW)
    assert load_labels(tmp_path, ["b", "a"])["X"] == Labels("B", {"From A"})


def test_a_package_not_cached_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not cached"):
        load_labels(tmp_path, ["missing"])


def test_a_package_whose_digest_moved_is_refused(tmp_path: Path) -> None:
    fetch_packages(FakeNse({A: label_package({})}), tmp_path, [("a", A)], NOW)
    (tmp_path / "a.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_labels(tmp_path, ["a"])
