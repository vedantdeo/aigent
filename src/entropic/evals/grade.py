"""The grading vocabulary, and the graders that cost nothing.

`Outcome` and `Score` live here rather than with the runner so a grader never imports it. Graders
are factories: `exact_match("company")` returns the grader, so the runner treats them all alike. The
paid one, LLM-as-judge, is in `judge.py`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from anthropic.types import Usage
from pydantic import BaseModel, JsonValue, ValidationError

from entropic.evals.dataset import Case


@dataclass(frozen=True)
class Outcome:
    """What a task produced for one case.

    `usage` is None for a task that never called the API.
    """

    output: dict[str, JsonValue] = field(default_factory=dict)
    usage: Usage | None = None
    model: str | None = None
    raw: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Score:
    """One grader's verdict on one outcome.

    `parts` carries field-level detail; `value` carries a number, for a grader whose verdict is one.
    `usage` is set only by graders that spend.
    """

    passed: bool
    detail: str = ""
    parts: Mapping[str, bool] = field(default_factory=dict)
    value: float | None = None
    usage: Usage | None = None
    model: str | None = None


Grader = Callable[[Case, Outcome], Score]


def _comparable(value: JsonValue) -> str:
    """Text form used for equality: whitespace collapsed and case folded for strings."""
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return json_ish(value)


def _as_text(value: JsonValue) -> str:
    """Text form with nothing normalised away — what a pattern gets matched against.

    A regex already has `(?i)` and `\\s+` for these; normalising first would take that choice away.
    """
    return value if isinstance(value, str) else json_ish(value)


def json_ish(value: JsonValue) -> str:
    """Stable text for a non-string value, so lists and numbers compare sensibly."""
    if isinstance(value, list):
        return "[" + ", ".join(json_ish(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {json_ish(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, bool) or value is None:
        return str(value).lower()
    return str(value)


def _both_present(case: Case, outcome: Outcome, name: str) -> Score | None:
    """A missing label is a dataset bug; a missing output field is a failure. Say which."""
    if name not in case.expected:
        return Score(False, f"dataset has no expected {name!r} for this case")
    if name not in outcome.output:
        return Score(False, f"output has no {name!r} field")
    return None


def exact_match(name: str = "answer") -> Grader:
    """Output field equals the label, compared as text with whitespace and case normalised."""

    def grade(case: Case, outcome: Outcome) -> Score:
        problem = _both_present(case, outcome, name)
        if problem is not None:
            return problem
        want, got = case.expected[name], outcome.output[name]
        if _comparable(want) == _comparable(got):
            return Score(True)
        return Score(False, f"expected {json_ish(want)!r}, got {json_ish(got)!r}")

    return grade


def contains(name: str = "answer") -> Grader:
    """The label appears somewhere inside the output field. Case-insensitive."""

    def grade(case: Case, outcome: Outcome) -> Score:
        problem = _both_present(case, outcome, name)
        if problem is not None:
            return problem
        needle, haystack = _comparable(case.expected[name]), _comparable(outcome.output[name])
        if needle in haystack:
            return Score(True)
        return Score(False, f"{needle!r} not found in {haystack[:120]!r}")

    return grade


def regex(name: str = "answer", pattern: str | None = None) -> Grader:
    """Output field matches a pattern.

    Pass `pattern` for one rule across the dataset, or leave it out to take one per case from
    `expected[name]`. Neither side is normalised; write `(?i)` for case-insensitivity.
    """
    fixed = re.compile(pattern) if pattern is not None else None

    def grade(case: Case, outcome: Outcome) -> Score:
        if name not in outcome.output:
            return Score(False, f"output has no {name!r} field")
        rule = fixed
        if rule is None:
            per_case = case.expected.get(name)
            if not isinstance(per_case, str):
                return Score(False, f"dataset has no regex in expected {name!r}")
            try:
                rule = re.compile(per_case)
            except re.error as exc:
                return Score(False, f"invalid regex {per_case!r}: {exc}")
        text = _as_text(outcome.output[name])
        if rule.search(text):
            return Score(True)
        return Score(False, f"{rule.pattern!r} did not match {text[:120]!r}")

    return grade


def pydantic_valid(model: type[BaseModel]) -> Grader:
    """The output validates against a schema. Structure only, not correctness."""

    def grade(case: Case, outcome: Outcome) -> Score:
        del case
        if outcome.error is not None:
            return Score(False, f"task failed: {outcome.error}")
        try:
            model.model_validate(outcome.output)
        except ValidationError as exc:
            problem = exc.errors()[0]
            where = ".".join(str(part) for part in problem["loc"]) or "<output>"
            return Score(False, f"{where}: {problem['msg']}")
        return Score(True)

    return grade


def field_match(names: Sequence[str]) -> Grader:
    """Exact match per field, reported per field.

    `passed` is the strict whole-record number; the useful one is usually in `parts`.
    """
    fields = tuple(names)

    def grade(case: Case, outcome: Outcome) -> Score:
        parts: dict[str, bool] = {}
        wrong: list[str] = []
        for name in fields:
            want = case.expected.get(name)
            got = outcome.output.get(name)
            ok = (
                name in case.expected
                and name in outcome.output
                and _comparable(want) == _comparable(got)
            )
            parts[name] = ok
            if not ok:
                wrong.append(f"{name}: expected {json_ish(want)!r}, got {json_ish(got)!r}")
        return Score(all(parts.values()), "; ".join(wrong), parts=parts)

    return grade


# --- Retrieval -------------------------------------------------------------------------------
# Both read `outcome.output["retrieved"]` against `case.expected["relevant"]`, and both must
# de-duplicate the ranked list first — a chunk returned twice has been found once.


def recall_at_k(k: int, *, retrieved: str = "retrieved", relevant: str = "relevant") -> Grader:
    """What fraction of this case's relevant chunks came back in the top k.

    `passed` means every one of them; `value` is the fraction the report means. Refuses k below 1 at
    build time.
    """
    raise NotImplementedError


def reciprocal_rank(
    *, k: int | None = None, retrieved: str = "retrieved", relevant: str = "relevant"
) -> Grader:
    """One over the rank of the first relevant chunk, zero if none came back.

    Averaged over a dataset this is MRR. `k` truncates the list first, for an MRR@k matching what
    the generator will see.
    """
    raise NotImplementedError
