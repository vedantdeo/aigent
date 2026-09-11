"""Project 1a: the task, and the dataset it runs on.

Two kinds of test here. The task's behaviour is checked against a scripted client, so it costs
nothing. The dataset is checked against the schema's own vocabulary — a label that says `Q5` or
`rising` is a bug in the asset, and the asset is the part that is expensive to rebuild.
"""

from __future__ import annotations

import re
from typing import cast

import anthropic
import pytest
from anthropic.types import MessageTokensCount, Usage
from pydantic import JsonValue

from entropic.config import MAX_TOKENS_HEADLINE, MODEL
from entropic.evals.dataset import Case, load_jsonl
from entropic.evals.runner import Task
from entropic.week01.extraction import (
    DATASET,
    FEW_SHOT,
    FIELDS,
    METRICS,
    VARIANTS,
    ZERO_SHOT,
    Extraction,
    Resolver,
    extraction_task,
    load_directory,
    spread,
)

DIRECTORY = load_directory()

CASES = load_jsonl(DATASET)
LABELLED = [case for case in CASES if case.expected]

# NSE symbols are uppercase alphanumerics; M&M is the one that needs the ampersand.
TICKER = re.compile(r"[A-Z0-9&]+")

VOCABULARY: dict[str, set[str]] = {
    "metric": {"revenue", "profit", "margin", "orders", "headcount", "guidance"},
    "quarter": {"Q1", "Q2", "Q3", "Q4", "FY", "unknown"},
    "direction": {"up", "down", "flat", "unknown"},
}

RECORD = Extraction(
    company="Infosys", metric="profit", quarter="Q3", direction="up", change_pct=11.5
)


class _Parsed:
    def __init__(self, record: Extraction | None) -> None:
        self.parsed_output = record
        self.usage = Usage(input_tokens=320, output_tokens=60)
        self.stop_reason = "max_tokens" if record is None else "end_turn"


class _Messages:
    def __init__(self, record: Extraction | None) -> None:
        self._record = record
        self.systems: list[str] = []
        self.counted = 0

    def count_tokens(self, **_: object) -> MessageTokensCount:
        self.counted += 1
        return MessageTokensCount(input_tokens=320)

    def parse(self, **kwargs: object) -> _Parsed:
        self.systems.append(cast(str, kwargs["system"]))
        assert kwargs["max_tokens"] == MAX_TOKENS_HEADLINE
        return _Parsed(self._record)


class _Client:
    def __init__(self, record: Extraction | None) -> None:
        self.messages = _Messages(record)


def _task_and_log(
    record: Extraction | None = RECORD,
    system: str = ZERO_SHOT,
    resolver: Resolver | None = None,
) -> tuple[Task, _Messages]:
    fake = _Client(record)
    task = extraction_task(system, resolver, client=cast(anthropic.Anthropic, fake))
    return task, fake.messages


def _case(headline: str = "Infosys Q3 net profit rises 11.5%") -> Case:
    return Case.model_validate({"id": "hl-001", "input": {"headline": headline}})


def test_a_parsed_record_becomes_a_gradeable_outcome() -> None:
    task, log = _task_and_log()
    outcome = task(_case())

    assert outcome.error is None
    assert outcome.output["company"] == "Infosys", "the model's own words survive for debugging"
    assert outcome.output["ticker"] == "INFY", "resolved in code, not asked of the model"
    assert outcome.output["change_pct"] == 11.5
    assert set(FIELDS) <= set(outcome.output), "everything field_match grades is present"
    assert outcome.usage is not None and outcome.model == MODEL, "the row can be billed"
    assert log.counted == 1, "the paid call was counted first"


def test_a_refusal_to_parse_is_an_error_row_that_still_bills() -> None:
    """The model hit the cap mid-record. That is one lost row, not a lost run."""
    task, _ = _task_and_log(record=None)
    outcome = task(_case())

    assert outcome.error is not None and "max_tokens" in outcome.error
    assert outcome.usage is not None, "it still cost money, so the runner must still see usage"
    assert outcome.output == {}


def test_a_case_with_no_headline_never_reaches_the_api() -> None:
    task, log = _task_and_log()
    outcome = task(Case.model_validate({"id": "hl-999", "input": {"abstract": "wrong field"}}))

    assert outcome.error is not None and "hl-999" in outcome.error
    assert log.counted == 0, "no headline, no call"


def test_the_headline_is_what_travels_and_the_variant_is_the_system_prompt() -> None:
    task, log = _task_and_log(system=FEW_SHOT)
    task(_case("TCS Q2 revenue up 7.9% YoY"))

    assert log.systems == [FEW_SHOT]
    assert "Worked examples" in FEW_SHOT and "Worked examples" not in ZERO_SHOT


def test_the_arms_are_an_ablation_ladder() -> None:
    """Each arm differs from the one before it by exactly one thing, or the table reads as noise."""
    assert list(VARIANTS) == ["zero_shot", "few_shot"]
    assert FEW_SHOT.startswith(ZERO_SHOT), "few_shot adds examples and nothing else"


@pytest.mark.parametrize(
    ("mention", "ticker"),
    [
        ("Infosys", "INFY"),
        ("Infy", "INFY"),
        ("INFY", "INFY"),
        ("HUL", "HINDUNILVR"),
        ("L&T", "LT"),
        ("SBI", "SBIN"),
        ("Hero MotoCorp", "HEROMOTOCO"),
        ("infosys ltd", "INFY"),
        ("Infosys Limited", "INFY"),
        ("  Dr. Reddy's  ", "DRREDDY"),
        ("Reliance Jio", None),
        ("", None),
    ],
)
def test_the_resolver_is_exact_after_normalising_and_never_approximate(
    mention: str, ticker: str | None
) -> None:
    """Case, punctuation, spacing and a Ltd suffix are noise. Everything else is a miss, on
    purpose: a lookup that matches approximately is wrong in the same quiet way a guess is."""
    assert Resolver()(mention) == ticker


@pytest.mark.parametrize(
    "alias",
    sorted({a for entry in DIRECTORY.values() for a in (entry["name"], *entry["aliases"])}),
)
def test_every_name_the_directory_lists_resolves_to_its_own_ticker(alias: str) -> None:
    resolver = Resolver()
    assert resolver(alias) is not None, f"{alias!r} is in the directory but does not resolve"


def test_a_mention_the_directory_does_not_know_is_a_to_do_not_a_wrong_answer() -> None:
    resolver = Resolver()
    task, _ = _task_and_log(
        record=Extraction(
            company="Reliance Jio", metric="revenue", quarter="Q2", direction="up", change_pct=None
        ),
        resolver=resolver,
    )
    outcome = task(_case("Reliance Jio adds 3.4 million subscribers in September"))

    assert outcome.error is None, "the row ran; it is the directory that is short a line"
    assert outcome.output["ticker"] is None, "no ticker invented to fill the hole"
    assert outcome.output["company"] == "Reliance Jio"
    assert resolver.unresolved == {"Reliance Jio": 1}, "and it says what to add"


def test_the_directory_covers_every_ticker_the_dataset_labels() -> None:
    """The directory is the ground truth for `ticker`. A label outside it is unfalsifiable."""
    labelled = {str(case.expected["ticker"]) for case in LABELLED}
    assert labelled <= set(DIRECTORY), f"not in the directory: {sorted(labelled - set(DIRECTORY))}"


@pytest.mark.parametrize("case", LABELLED, ids=lambda case: case.id)
def test_every_headline_names_its_company_in_a_form_the_directory_lists(case: Case) -> None:
    """The alias the headline uses has to be resolvable, or the label is a leap the model cannot
    make and the directory cannot help with."""
    headline = str(case.input["headline"])
    entry = DIRECTORY[str(case.expected["ticker"])]
    assert any(alias.casefold() in headline.casefold() for alias in entry["aliases"]), (
        f"{case.id}: none of {entry['aliases']} appears in {headline!r}"
    )


@pytest.mark.parametrize("case", LABELLED, ids=lambda case: case.id)
def test_copying_the_headline_correctly_is_enough_to_get_the_ticker_right(case: Case) -> None:
    """The property the whole design rests on: the model only has to copy, and the ticker follows.

    For each case, take the alias the headline actually uses — what a careful model would return —
    and check the resolver lands on the label. If this holds for all 30, `ticker` is right by
    construction whenever `company` is, and the eval measures identification rather than recall.
    """
    headline = str(case.input["headline"]).casefold()
    expected = str(case.expected["ticker"])
    aliases = DIRECTORY[expected]["aliases"]
    as_written = max((a for a in aliases if a.casefold() in headline), key=len)

    assert Resolver()(as_written) == expected


def test_the_metric_order_is_total_and_covers_the_vocabulary() -> None:
    """A headline routinely names two metrics. Without a total order the label is a coin flip the
    model has no way to call, and the miss reads as a model failure when it is a spec failure."""
    assert set(METRICS) == VOCABULARY["metric"], "every metric is ranked, and nothing extra is"
    assert len(set(METRICS)) == len(METRICS), "a total order has no ties"

    description = Extraction.model_fields["metric"].description or ""
    assert ", ".join(METRICS) in description, (
        "the model is shown the order, not just told there is one"
    )
    # Without this, the mechanical rule reads "cuts revenue guidance" as cueing revenue, which
    # outranks guidance and would flip hl-003 and hl-023 to a label no reader would write.
    assert "forward-looking" in description and "guidance, not that metric" in description


def test_both_variants_state_the_conventions_the_dataset_is_labelled_by() -> None:
    """A rule that only appears in the few-shot examples would punish zero_shot for not guessing."""
    descriptions = " ".join(field.description or "" for field in Extraction.model_fields.values())
    for convention in ("verbatim", "June quarter is Q1", "basis", "unknown"):
        assert convention in descriptions, convention


# --- the dataset is an asset, so it gets tested like one ----------------------------------------


@pytest.mark.parametrize(
    ("n", "want"),
    [
        (4, ["hl-001", "hl-008", "hl-015", "hl-022"]),
        (1, ["hl-001"]),
        (30, [case.id for case in LABELLED]),
        (99, [case.id for case in LABELLED]),
    ],
)
def test_a_sample_spans_the_dataset_rather_than_taking_the_front(n: int, want: list[str]) -> None:
    """The rows are roughly in the order they were written, so the first four are four easy ones.
    A smoke run that cannot fail is not worth its money."""
    assert [case.id for case in spread(LABELLED, n)] == want


def test_the_dataset_is_the_size_the_project_asked_for() -> None:
    assert 40 <= len(CASES) <= 60, "Project 1a: 40 to 60 inputs"
    assert len(LABELLED) >= 30, "30 of them hand-labelled"


@pytest.mark.parametrize("case", LABELLED, ids=lambda case: case.id)
def test_every_label_validates_against_the_vocabulary(case: Case) -> None:
    # The label carries `ticker`, which the model never produces — it is resolved in code — so the
    # label is validated against `Extraction` with the mention filled in from the directory.
    ticker = str(case.expected["ticker"])
    record = Extraction.model_validate({**case.expected, "company": DIRECTORY[ticker]["name"]})

    assert set(case.expected) == set(FIELDS), "a label with a missing field grades as wrong forever"
    assert TICKER.fullmatch(ticker), f"{case.id}: {ticker!r} is not a ticker"
    for name, allowed in VOCABULARY.items():
        assert getattr(record, name) in allowed, f"{case.id}: {name}={getattr(record, name)!r}"
    if record.change_pct is not None:
        assert record.change_pct > 0, "change_pct is a magnitude; direction carries the sign"
        assert isinstance(case.expected["change_pct"], float), (
            "label floats, not ints: field_match compares text and str(12) != str(12.0)"
        )


def test_the_labels_do_not_let_a_constant_answer_score_well() -> None:
    """A field every row agrees on measures nothing.

    This is a check on the dataset, not on the model: it fails when the labels stop discriminating.
    """
    for name in FIELDS:
        values = [cast(JsonValue, case.expected[name]) for case in LABELLED]
        commonest = max(set(map(str, values)), key=lambda v: list(map(str, values)).count(v))
        share = list(map(str, values)).count(commonest) / len(values)
        assert share <= 0.6, f"{name}: always answering {commonest!r} would score {share:.0%}"
