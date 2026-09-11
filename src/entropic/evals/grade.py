"""The grading vocabulary, and the four graders that cost nothing.

`Outcome` and `Score` both live here rather than next to the runner so that a grader never has to
import the runner — the dependency runs one way, graders → vocabulary, and stays acyclic.

Four graders are pure functions of (case, outcome): exact match, contains, regex, and Pydantic
validity, plus `field_match` for the per-field accuracy an extraction task is actually judged on.
The fifth grader, LLM-as-judge, spends money and lives in `judge.py`.

Graders are factories: `exact_match("company")` returns the grader. That keeps the callable
signature uniform, which is what lets the runner treat every grader the same way.
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

    `usage` is None for a task that never called the API — a retrieval eval, or a fake used in
    tests. The runner bills whatever is here and nothing more.
    """

    output: dict[str, JsonValue] = field(default_factory=dict)
    usage: Usage | None = None
    model: str | None = None
    raw: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Score:
    """One grader's verdict on one outcome.

    `parts` carries field-level detail when a grader has it; the report pivots it into a per-field
    table. `usage` is set only by graders that spend (the judge), and the runner bills it to the
    same eval budget as the task.
    """

    passed: bool
    detail: str = ""
    parts: Mapping[str, bool] = field(default_factory=dict)
    usage: Usage | None = None
    model: str | None = None


Grader = Callable[[Case, Outcome], Score]


def _comparable(value: JsonValue) -> str:
    """Text form used for equality: whitespace collapsed and case folded for strings."""
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return json_ish(value)


def _as_text(value: JsonValue) -> str:
    """Text form with nothing normalised away. What a pattern gets matched against.

    A regex is a specification, not a value: it already has `(?i)` for case and `\\s+` for loose
    whitespace. Normalising the subject first would take those decisions away from whoever wrote the
    pattern — and quietly make `^[A-Z]+$` unsatisfiable.
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

    Pass `pattern` for one rule across the whole dataset; leave it out to take the pattern from each
    case's `expected[name]`, which is how you check a format that varies per row.

    Both sides are used as written: the value is not normalised the way `exact_match` normalises it,
    and the pattern is certainly not — casefolding a pattern turns `\\D` into `\\d` and inverts the
    check. Write `(?i)` into the pattern when you want case-insensitivity.
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
    """The output validates against a schema. Structure only — it says nothing about correctness.

    This is the grader that answers "did the model return the shape I asked for", which is a
    different question from "is the content right", and worth keeping separate.
    """

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

    `passed` means every field was right, which is the strict per-record number. The useful number
    is usually in `parts`: whole-record accuracy on a five-field schema reads near zero and tells
    you nothing about which field is the problem.
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
