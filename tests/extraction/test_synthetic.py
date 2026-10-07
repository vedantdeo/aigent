"""The toy fine-tune's training set: labels right by construction, and none quoting the test set.

Free and offline: nothing is sent, and rendering a row needs only the wire's message format.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import cast

import pytest

from aigent.adapters import spec
from aigent.adapters.openai import record_in
from aigent.config import SYNTHETIC_CLIENT, SYNTHETIC_HEADLINES, max_tokens
from aigent.extraction import synthetic
from aigent.extraction.headlines import (
    METRICS,
    VARIANTS,
    Extraction,
    Resolver,
    headline_request,
    word_runs,
)
from aigent.extraction.synthetic import (
    TEMPLATES,
    Synthetic,
    generate,
    preferred,
    real_headlines,
    render,
    split,
    wire_for,
)

EXAMPLES = generate()
QUARTERS = ("Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY", "unknown")
DIRECTIONS = ("up", "down", "flat", "unknown")


def test_the_set_is_the_size_asked_for_and_every_template_is_in_it() -> None:
    assert len(EXAMPLES) == SYNTHETIC_HEADLINES
    assert len({example.headline.casefold() for example in EXAMPLES}) == len(EXAMPLES)
    rules = {template(random.Random(0), "Acme").rule for template in TEMPLATES}
    assert {example.rule for example in EXAMPLES} == rules


def test_no_synthetic_headline_quotes_a_real_one() -> None:
    """The real headlines are the test set; a template that copied one would be tested on itself."""
    real = set[tuple[str, ...]]().union(*(word_runs(headline) for headline in real_headlines()))
    leaked = {
        e.headline: word_runs(e.headline) & real for e in EXAMPLES if word_runs(e.headline) & real
    }
    assert not leaked, leaked


def test_every_label_is_in_its_fields_vocabulary() -> None:
    wrong = [
        e.headline
        for e in EXAMPLES
        if e.label.metric not in METRICS
        or e.label.quarter not in QUARTERS
        or e.label.direction not in DIRECTIONS
        or (e.label.change_pct is not None and e.label.change_pct <= 0)
    ]
    assert not wrong, wrong


def test_the_company_is_copied_from_the_headline_and_resolves_to_a_ticker() -> None:
    resolver = Resolver()
    for example in EXAMPLES:
        company = example.label.company
        assert company is not None and example.headline.startswith(company), example.headline
        assert resolver(company) is not None, company


@pytest.mark.parametrize(
    ("rule", "holds"),
    [
        pytest.param(
            "margin-level",
            lambda e: e.label.metric == "margin" and e.label.change_pct is None,
            id="a margin moving between levels states no change",
        ),
        pytest.param(
            "basis-points",
            lambda e: e.label.metric == "margin" and e.label.change_pct is None,
            id="basis points are not a percentage",
        ),
        pytest.param(
            "rupee-amount",
            lambda e: e.label.change_pct is None,
            id="a rupee figure or a doubling is not a percentage",
        ),
        pytest.param(
            "flat",
            lambda e: e.label.direction == "flat" and e.label.change_pct is None,
            id="flat at a level has no change either",
        ),
        pytest.param(
            "guidance",
            lambda e: (e.label.metric, e.label.quarter) == ("guidance", "FY"),
            id="a forward-looking figure is guidance for the year",
        ),
        pytest.param(
            "orders",
            lambda e: (e.label.metric, e.label.direction) == ("orders", "unknown"),
            id="an order win has no direction",
        ),
        pytest.param(
            "growth-rate",
            lambda e: e.label.direction == "up" and f"to {e.label.change_pct:g}%" in e.headline,
            id="slower growth is still up, at the new rate",
        ),
        pytest.param(
            "share-price",
            lambda e: e.label.metric == "none" and e.label.change_pct is None,
            id="a share price move is no metric",
        ),
        pytest.param(
            "no-metric",
            lambda e: (e.label.metric, e.label.direction) == ("none", "unknown"),
            id="an announcement with no figure has no metric",
        ),
        pytest.param(
            "no-period",
            lambda e: e.label.quarter == "unknown",
            id="a bare month is not a quarter",
        ),
        pytest.param(
            "loss",
            lambda e: (
                e.label.metric == "profit"
                and (e.label.direction == "up") == ("narrows" in e.headline)
            ),
            id="a narrowing loss is profit going up",
        ),
    ],
)
def test_each_template_labels_by_the_rule_it_exercises(
    rule: str, holds: Callable[[Synthetic], bool]
) -> None:
    rows = [example for example in EXAMPLES if example.rule == rule]
    assert rows, f"no {rule} rows"
    wrong = [example.headline for example in rows if not holds(example)]
    assert not wrong, wrong


@pytest.mark.parametrize(
    ("stated", "mentioned", "label"),
    [
        pytest.param(["profit", "revenue"], ["profit", "revenue"], "revenue", id="both stated"),
        pytest.param(
            ["other"], ["margin", "other"], "other", id="only one stated, whatever its rank"
        ),
        pytest.param([], ["margin", "profit"], "profit", id="none stated: best ranked mentioned"),
    ],
)
def test_two_metrics_are_labelled_by_the_preference_order(
    stated: list[str], mentioned: list[str], label: str
) -> None:
    assert preferred(stated, mentioned) == label


def test_a_seed_makes_the_set_repeatable() -> None:
    def headlines(seed: int) -> list[str]:
        return [example.headline for example in generate(20, seed=seed)]

    assert headlines(7) == headlines(7)
    assert headlines(7) != headlines(8)


def test_the_split_keeps_a_tenth_back_and_shares_no_row() -> None:
    train, valid = split(EXAMPLES, 0.1, seed=0)
    assert (len(train), len(valid)) == (180, 20)
    assert not {example.headline for example in train} & {example.headline for example in valid}


def test_a_training_row_is_the_request_the_eval_sends_then_the_label() -> None:
    """Tuned on one prompt and asked with another, the eval would measure the difference."""
    wire = wire_for(SYNTHETIC_CLIENT)
    cap = max_tokens("HEADLINE", spec(SYNTHETIC_CLIENT))
    example = EXAMPLES[0]

    row = render(example, wire, cap)["messages"]

    request = headline_request("any", example.headline, VARIANTS["zero_shot"], cap)
    assert row[:-1] == [dict(message) for message in wire.messages(request, Extraction)]
    assert row[-1]["role"] == "assistant"
    assert record_in(str(row[-1]["content"]), Extraction) == example.label


def test_rows_are_only_rendered_for_the_wire_mlx_lm_trains_on() -> None:
    with pytest.raises(ValueError, match="openai"):
        wire_for("anthropic")


class _Draws:
    """Stands in for `random.Random` where a test needs one particular draw."""

    def __init__(self, value: int) -> None:
        self.value = value

    def randint(self, low: int, high: int) -> int:
        del low, high
        return self.value


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (120, "Rs 120 crore"),
        (4_567, "Rs 4,567 crore"),
        (99_999, "Rs 99,999 crore"),
        pytest.param(1_234_567, "Rs 12,34,567 crore", id="lakhs grouped in pairs"),
        pytest.param(123_456_789, "Rs 12,34,56,789 crore", id="crores grouped in pairs too"),
    ],
)
def test_a_crore_amount_is_grouped_the_indian_way(value: int, written: str) -> None:
    assert synthetic._crore(cast(random.Random, _Draws(value))) == written


def test_a_margin_never_moves_to_the_level_it_started_at(monkeypatch: pytest.MonkeyPatch) -> None:
    levels = iter([12.5, 12.5, 14.0])  # the first draw for `now` ties `before` and is drawn again
    monkeypatch.setattr(synthetic, "_level", lambda rng: next(levels))

    made = synthetic.margin_level(random.Random(0), "Infosys")

    assert "to 14% from 12.5%" in made.headline, made.headline


def test_templates_that_cannot_make_enough_distinct_headlines_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same = synthetic.margin_level(random.Random(0), "Infosys")
    monkeypatch.setattr(synthetic, "TEMPLATES", (lambda rng, company: same,))

    with pytest.raises(RuntimeError, match="made only 1 of 3 headlines"):
        synthetic.generate(3, real=[])
