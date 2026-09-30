"""Project 1a's classical baseline: what each arm is trained on, and what it may not see.

Free and offline: every model here is a scikit-learn fit on a few hundred headlines.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from pydantic import JsonValue
from sklearn.pipeline import Pipeline

from aigent.config import SYNTHETIC_HEADLINES, SYNTHETIC_VALID_FRACTION
from aigent.evals.dataset import Case
from aigent.evals.runner import Task
from aigent.extraction import baseline
from aigent.extraction.baseline import (
    FIELD,
    fit,
    leave_one_out,
    majority,
    synthetic_training,
    top_terms,
    trained_on,
)

HEADLINES, LABELS = synthetic_training()


def case(name: str, headline: str | None, direction: str = "up") -> Case:
    fields: dict[str, JsonValue] = {} if headline is None else {"headline": headline}
    return Case(id=name, input=fields, expected={FIELD: direction})


def test_the_template_arm_trains_on_the_fine_tunes_own_split() -> None:
    held_back = max(1, round(SYNTHETIC_HEADLINES * SYNTHETIC_VALID_FRACTION))
    assert len(HEADLINES) == len(LABELS) == SYNTHETIC_HEADLINES - held_back
    assert set(LABELS) == {"up", "down", "flat", "unknown"}


@pytest.mark.parametrize(
    ("headline", "direction"),
    [
        pytest.param("Acme Q2 net profit rises 12%", "up", id="a rise is up"),
        pytest.param("Acme Q3 revenue falls 8%", "down", id="a fall is down"),
        pytest.param("Acme Q1 revenue unchanged at Rs 1,200 crore", "flat", id="unchanged is flat"),
        pytest.param("Acme appoints new chief executive", "unknown", id="no metric has no way"),
    ],
)
def test_the_template_model_reads_the_rules_its_templates_teach(
    headline: str, direction: str
) -> None:
    outcome = trained_on(HEADLINES, LABELS)(case("t", headline))
    assert outcome.output[FIELD] == direction, outcome.output


def test_leave_one_out_never_trains_on_the_case_it_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def spy(headlines: Sequence[str], labels: Sequence[str]) -> Pipeline:
        seen.append(list(headlines))
        return fit(HEADLINES, LABELS)

    monkeypatch.setattr(baseline, "fit", spy)
    cases = [case(f"c{i}", f"Acme Q{i} revenue rises {i}%") for i in range(1, 4)]
    task = leave_one_out(cases)
    for asked in cases:
        task(asked)
        assert asked.input["headline"] not in seen[-1], seen[-1]
        assert len(seen[-1]) == len(cases) - 1


@pytest.mark.parametrize(
    ("labels", "answers"),
    [
        pytest.param(
            ("down", "up", "up", "up"),
            ("up", "up", "up", "up"),
            id="the lone dissenter is outvoted",
        ),
        pytest.param(
            ("down", "down", "down", "up"),
            ("down", "down", "down", "down"),
            id="the commonest label wins even when it is not up",
        ),
    ],
)
def test_majority_answers_the_commonest_label_among_the_others(
    labels: tuple[str, ...], answers: tuple[str, ...]
) -> None:
    cases = [case(f"c{i}", "Acme", label) for i, label in enumerate(labels)]
    task = majority(cases)
    got = tuple(task(c).output[FIELD] for c in cases)
    assert got == answers, got


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda cs: trained_on(HEADLINES, LABELS), id="the template arm"),
        pytest.param(leave_one_out, id="the leave-one-out arm"),
    ],
)
def test_a_case_without_a_headline_is_an_error_row(
    make: Callable[[Sequence[Case]], Task],
) -> None:
    cases = [case("a", "Acme revenue rises 4%"), case("b", "Acme revenue falls 4%", "down")]
    task = make(cases)
    outcome = task(case("missing", None))
    assert outcome.error is not None and "missing" in outcome.error


def test_top_terms_names_terms_for_every_label() -> None:
    terms = top_terms(fit(HEADLINES, LABELS), k=3)
    assert set(terms) == set(LABELS)
    assert all(len(words) == 3 for words in terms.values()), terms
