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
# Both read `outcome.output["retrieved"]` against `case.expected["relevant"]` — different keys on
# each side, which is why `_both_present` above does not fit.


def _id_list(value: JsonValue) -> list[str] | None:
    """A list of chunk ids, order kept and repeats dropped. None if it is not that shape.

    A chunk returned twice has been found once, and would otherwise occupy two of the k slots.
    """
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(dict.fromkeys(item for item in value if isinstance(item, str)))


def _relevant_ids(case: Case, relevant: str) -> list[str] | Score:
    """This case's labelled chunk ids, or the Score saying why the label cannot be used.

    An empty list is the interesting one: the labelled quote is in no single chunk under the
    strategy being scored, so nothing can retrieve it. Shared with `resolvable`, which reports how
    often that happens.
    """
    if relevant not in case.expected:
        return Score(False, f"dataset has no expected {relevant!r} for this case")
    want = _id_list(case.expected[relevant])
    if want is None:
        return Score(False, f"expected {relevant!r} is not a list of chunk ids")
    if not want:
        return Score(False, "the labelled quote is in no single chunk under this strategy")
    return want


def resolvable(*, relevant: str = "relevant") -> Grader:
    """Whether this case's label survived chunking — a property of the splitter, not the retriever.

    A quote no single chunk contains cannot be found by any retriever, so the row is a chunking
    failure wearing a retrieval failure's clothes. This is the denominator the other two metrics
    are meaned over: `recall_at_k` and `reciprocal_rank` score nothing on such a row, so their
    columns mean less than they appear to without this one beside them.
    """

    def grade(case: Case, outcome: Outcome) -> Score:
        want = _relevant_ids(case, relevant)
        if isinstance(want, Score):
            return Score(False, want.detail, value=0.0)
        return Score(True, "", value=1.0)

    return grade


def _ranked_and_relevant(
    case: Case, outcome: Outcome, retrieved: str, relevant: str
) -> tuple[list[str], list[str]] | Score:
    """Both id lists, or the Score saying which side is unusable.

    A label that is missing, malformed or empty is a dataset problem; a missing or malformed ranked
    list is a task failure. Shared, so both graders answer the same way.
    """
    want = _relevant_ids(case, relevant)
    if isinstance(want, Score):
        return want
    if retrieved not in outcome.output:
        return Score(False, f"output has no {retrieved!r} field")
    got = _id_list(outcome.output[retrieved])
    if got is None:
        return Score(False, f"output {retrieved!r} is not a list of chunk ids")
    return got, want


def recall_at_k(k: int, *, retrieved: str = "retrieved", relevant: str = "relevant") -> Grader:
    """What fraction of this case's relevant chunks came back in the top k.

    `passed` means every one of them; `value` is the fraction the report means. Refuses k below 1 at
    build time.
    """
    if k < 1:
        raise ValueError(f"recall_at_k: k must be at least 1, got {k}")

    def grade(case: Case, outcome: Outcome) -> Score:
        both = _ranked_and_relevant(case, outcome, retrieved, relevant)
        if isinstance(both, Score):
            return both
        got, want = both
        top = set(got[:k])
        missed = [chunk_id for chunk_id in want if chunk_id not in top]
        found = len(want) - len(missed)
        detail = "" if not missed else f"{found}/{len(want)} in top {k}; missed {', '.join(missed)}"
        return Score(not missed, detail, value=found / len(want))

    return grade


def hit_at_k(k: int, *, retrieved: str = "retrieved", relevant: str = "relevant") -> Grader:
    """Whether any relevant chunk came back in the top k — can the generator answer at all.

    `recall_at_k` asks for all of them, which charges a strategy for the overlap that made the
    quote resolve to several chunks in the first place. Those extra chunks are the same sentence
    seen through a second window, not a second fact, so one of them is a complete answer.
    """
    if k < 1:
        raise ValueError(f"hit_at_k: k must be at least 1, got {k}")

    def grade(case: Case, outcome: Outcome) -> Score:
        both = _ranked_and_relevant(case, outcome, retrieved, relevant)
        if isinstance(both, Score):
            return both
        got, want = both
        hit = next((chunk_id for chunk_id in got[:k] if chunk_id in want), None)
        if hit is None:
            return Score(False, f"none of {len(want)} relevant chunks in top {k}", value=0.0)
        return Score(True, "", value=1.0)

    return grade


def reciprocal_rank(
    *, k: int | None = None, retrieved: str = "retrieved", relevant: str = "relevant"
) -> Grader:
    """One over the rank of the first relevant chunk, zero if none came back.

    Averaged over a dataset this is MRR. `k` truncates the list first, for an MRR@k matching what
    the generator will see.
    """
    if k is not None and k < 1:
        raise ValueError(f"reciprocal_rank: k must be at least 1, got {k}")

    def grade(case: Case, outcome: Outcome) -> Score:
        both = _ranked_and_relevant(case, outcome, retrieved, relevant)
        if isinstance(both, Score):
            return both
        got, want = both
        if k is not None:
            got = got[:k]
        for rank, chunk_id in enumerate(got, start=1):
            if chunk_id in want:
                return Score(
                    rank == 1,
                    "" if rank == 1 else f"first relevant chunk at rank {rank} ({chunk_id})",
                    value=1 / rank,
                )
        return Score(False, "no relevant chunk found", value=0.0)

    return grade
