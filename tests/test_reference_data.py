"""Hand-edited reference files: sorted by their key, and no key twice.

`ORDERED_FILES` is the opt-in registry — add a row when a new file lands. Opt-in rather than a sweep
because a sequence whose order is meaningful (`extraction.METRICS`) must not be "fixed".
See `CLAUDE.md`, "Reference data stays in one canonical order".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from aigent.extraction.headlines import DIRECTORY
from aigent.retrieval.corpus import MANIFEST
from aigent.retrieval.questions import DATASET

PARAPHRASED = DATASET.with_name("retrieval-paraphrased.jsonl")

Pairs = list[tuple[str, object]]


def json_keys(*into: str) -> Callable[[Path], list[str]]:
    """Read a JSON file and return the keys of one nested object, exactly as the file writes them.

    Parsed through `object_pairs_hook=list`, so keys arrive in file order **and duplicates survive**
    — a plain `json.loads` builds a dict, which keeps the last of a repeated key and silently drops
    the rest, and a check that runs after that can never see the thing it is looking for.
    """

    def read(path: Path) -> list[str]:
        node = cast(Pairs, json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=list))
        for step in into:
            node = cast(Pairs, next(value for key, value in node if key == step))
        return [key for key, _ in node]

    return read


def jsonl_ids(field: str) -> Callable[[Path], list[str]]:
    """Read one field from every line of a JSONL file, in file order."""

    def read(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [str(json.loads(line)[field]) for line in lines if line.strip()]

    return read


@dataclass(frozen=True)
class Ordered:
    """A hand-edited file whose entries stay sorted by the key they are looked up by."""

    path: Path
    key: str
    load: Callable[[Path], list[str]]

    def __str__(self) -> str:
        return self.path.name


ORDERED_FILES = [
    Ordered(DIRECTORY, key="ticker", load=json_keys("companies")),
    Ordered(MANIFEST, key="doc_id", load=json_keys("documents")),
    Ordered(DATASET, key="id", load=jsonl_ids("id")),
    Ordered(PARAPHRASED, key="id", load=jsonl_ids("id")),
]


@pytest.mark.parametrize("ordered", ORDERED_FILES, ids=str)
def test_reference_data_is_sorted_by_its_key(ordered: Ordered) -> None:
    """Sort on write, in whatever dumps the file. This is the guard, not the mechanism."""
    keys = ordered.load(ordered.path)
    out_of_place = [k for k, s in zip(keys, sorted(keys), strict=True) if k != s]

    assert keys == sorted(keys), (
        f"{ordered.path.name} is not sorted by {ordered.key}; first out of place: "
        f"{out_of_place[:3]}. Re-dump it sorted by its key."
    )


@pytest.mark.parametrize("ordered", ORDERED_FILES, ids=str)
def test_reference_data_has_no_repeated_key(ordered: Ordered) -> None:
    """A duplicate key is invisible after parsing — JSON keeps the last one and drops the rest — so
    an entry can be edited and have no effect. Catch it while both copies still exist."""
    keys = ordered.load(ordered.path)
    repeated = sorted({k for k in keys if keys.count(k) > 1})

    assert not repeated, f"{ordered.path.name}: {ordered.key} appears more than once: {repeated}"
