"""Eval datasets: one JSONL file, one `Case` per line.

{"id": "hl-001", "input": {"headline": "..."}, "expected": {"company": "..."}, "tags": ["q3"]}

The loader is strict: unknown keys, duplicate ids and malformed lines all fail here, with the line
number, before the first paid call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError


class DatasetError(ValueError):
    """A dataset that cannot be trusted: bad JSON, an invalid row, or duplicate ids."""


class Case(BaseModel):
    """One row of an eval dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Stable across runs; it is how you cite a failure.")
    input: dict[str, JsonValue] = Field(description="What the task under test receives.")
    expected: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="The label. Empty for cases that are only checked for validity.",
    )
    tags: list[str] = Field(default_factory=list, description="For slicing the results table.")


def load_jsonl(path: Path) -> list[Case]:
    """Read and validate a whole dataset, raising DatasetError with a line number on a bad row.

    Blank lines and `//` comment lines are skipped.
    """
    cases: list[Case] = []
    first_seen: dict[str, int] = {}

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}:{lineno}: not valid JSON ({exc.msg})") from exc
        try:
            case = Case.model_validate(raw)
        except ValidationError as exc:
            raise DatasetError(f"{path.name}:{lineno}: {_first_problem(exc)}") from exc
        if case.id in first_seen:
            raise DatasetError(
                f"{path.name}:{lineno}: duplicate id {case.id!r}, already used on line "
                f"{first_seen[case.id]}"
            )
        first_seen[case.id] = lineno
        cases.append(case)

    if not cases:
        raise DatasetError(f"{path.name}: no cases found")
    return cases


def _first_problem(exc: ValidationError) -> str:
    """The first validation error, phrased for someone looking at a line of JSON."""
    problem = exc.errors()[0]
    where = ".".join(str(part) for part in problem["loc"]) or "<row>"
    return f"{where}: {problem['msg']}"


def digest(path: Path) -> str:
    """Short SHA-256 of the dataset file, recorded in every report.

    Without it you cannot tell whether a score moved because the prompt changed or the data did.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
