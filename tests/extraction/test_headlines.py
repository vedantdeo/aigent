"""Project 1a: the task, and the dataset it runs on.

Two kinds of test here. The task's behaviour is checked against a scripted client, so it costs
nothing. The dataset is checked against the schema's own vocabulary — a label that says `Q5` or
`rising` is a bug in the asset, and the asset is the part that is expensive to rebuild.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import cast

import pytest
from anthropic.types import MessageTokensCount, Usage
from pydantic import JsonValue

from aigent.config import CLIENT, MAX_TOKENS_HEADLINE, MODEL, THINKING_EVAL
from aigent.evals.dataset import Case, load_jsonl
from aigent.evals.runner import Task
from aigent.extraction.headlines import (
    CACHE_VARIANTS,
    DATASET,
    FEW_SHOT,
    FIELDS,
    METRICS,
    THINKING,
    VARIANTS,
    ZERO_SHOT,
    Extraction,
    Resolver,
    Variant,
    _key,
    extraction_task,
    load_directory,
    spread,
    word_runs,
)

from ..conftest import CAPPED, ParsedReply

DIRECTORY = load_directory()


def written_as(case: Case) -> str:
    """The longest spelling of the case's company that its headline actually contains."""
    headline = str(case.input["headline"]).casefold()
    ticker = str(case.expected["ticker"])
    return max((f for f in forms(ticker) if f.casefold() in headline), key=len)


def shadowed_by(spelling: str, ticker: str) -> list[str]:
    """Other companies whose own spelling sits whole inside this one."""
    return sorted(
        {
            other
            for other in DIRECTORY
            if other != ticker
            for form in forms(other)
            if len(form) < len(spelling) and re.search(rf"\b{re.escape(form)}\b", spelling, re.I)
        }
    )


def forms(ticker: str) -> list[str]:
    """Every spelling the directory knows for one company: the symbol, the registered name, then
    the aliases — which hold only what the first two do not already say."""
    entry = DIRECTORY[ticker]
    return [ticker, entry["name"], *entry["aliases"]]


CASES = load_jsonl(DATASET)
LABELLED = [case for case in CASES if case.expected]
# A headline naming no single company has no ticker, so the company checks skip it.
NAMED = [case for case in LABELLED if case.expected["ticker"] is not None]

# Uppercase, plus the & in M&M and the hyphen in BAJAJ-AUTO.
TICKER = re.compile(r"[A-Z0-9&-]+")

VOCABULARY: dict[str, set[str]] = {
    "metric": {"revenue", "profit", "margin", "orders", "headcount", "guidance", "other", "none"},
    "quarter": {"Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY", "unknown"},
    "direction": {"up", "down", "flat", "unknown"},
}

PLAIN = Variant(ZERO_SHOT)

RECORD = Extraction(
    company="Infosys", metric="profit", quarter="Q3", direction="up", change_pct=11.5
)


class _Messages:
    def __init__(self, record: Extraction | None) -> None:
        self._record = record
        self.systems: list[object] = []
        self.thinking: list[object] = []
        self.caps: list[int] = []
        self.counted = 0

    def count_tokens(self, **_: object) -> MessageTokensCount:
        self.counted += 1
        return MessageTokensCount(input_tokens=320)

    def parse(self, **kwargs: object) -> ParsedReply:
        self.systems.append(kwargs["system"])
        self.thinking.append(kwargs["thinking"])
        self.caps.append(cast(int, kwargs["max_tokens"]))
        return ParsedReply(
            self._record,
            stop_reason="max_tokens" if self._record is None else "end_turn",
            usage=Usage(input_tokens=320, output_tokens=60),
        )


class _Client:
    def __init__(self, record: Extraction | None) -> None:
        self.messages = _Messages(record)


def _task_and_log(
    record: Extraction | None = RECORD,
    variant: Variant = PLAIN,
    resolver: Resolver | None = None,
    client: str = CLIENT,
) -> tuple[Task, _Messages]:
    fake = _Client(record)
    task = extraction_task(variant, resolver, client=client, sdk=fake)
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


@pytest.mark.usefixtures("capped")
@pytest.mark.parametrize(
    ("client", "cap"),
    [
        pytest.param(CLIENT, MAX_TOKENS_HEADLINE, id="the configured client's own cap"),
        pytest.param(CAPPED.name, CAPPED.max_tokens["HEADLINE"], id="a client named for the run"),
    ],
)
def test_a_record_is_capped_by_the_client_that_gives_it(client: str, cap: int) -> None:
    """`MAX_TOKENS_HEADLINE` names the configured client's cap, so a run pointed at another client
    has to ask for that client's own."""
    task, log = _task_and_log(client=client)

    task(_case())

    assert log.caps == [cap], f"sent on {client}"


def test_a_refusal_to_parse_is_an_error_row_that_still_bills() -> None:
    """The model hit the cap mid-record. That is one lost row, not a lost run."""
    task, _ = _task_and_log(record=None)
    outcome = task(_case())

    assert outcome.error is not None and "max_tokens" in outcome.error
    assert outcome.usage is not None, "it still cost money, so the runner must still see usage"
    assert outcome.output == {}


def test_an_eval_run_turns_extended_thinking_off() -> None:
    """The only determinism knob Claude 5 still offers. temperature and top_p are deprecated on
    the whole family, and thinking on its own made three identical calls return 192, 88 and 203
    output tokens and two different records — plus it counts against max_tokens, so it clipped
    two rows of a 50-row eval by spending the budget before reaching the answer."""
    assert THINKING_EVAL is False, "an eval is a measurement, not a conversation"
    assert THINKING == {"type": "disabled"}

    task, log = _task_and_log()
    task(_case())
    assert log.thinking == [{"type": "disabled"}], "and it has to reach the wire"


def test_the_breakpoint_reaches_the_wire() -> None:
    """The whole saving rides on this one field arriving. A `Variant` that built the block form
    but never sent it would look identical in every test that checks the prompt text."""
    task, log = _task_and_log(variant=Variant(FEW_SHOT, cache=True))
    task(_case())

    assert log.systems == [
        [{"type": "text", "text": FEW_SHOT, "cache_control": {"type": "ephemeral"}}]
    ]


def test_a_case_with_no_headline_never_reaches_the_api() -> None:
    task, log = _task_and_log()
    outcome = task(Case.model_validate({"id": "hl-999", "input": {"abstract": "wrong field"}}))

    assert outcome.error is not None and "hl-999" in outcome.error
    assert log.counted == 0, "no headline, no call"


def test_the_headline_is_what_travels_and_the_variant_is_the_system_prompt() -> None:
    task, log = _task_and_log(variant=Variant(FEW_SHOT))
    task(_case("TCS Q2 revenue up 7.9% YoY"))

    assert log.systems == [FEW_SHOT]
    assert "Worked examples" in FEW_SHOT and "Worked examples" not in ZERO_SHOT


def test_the_arms_are_an_ablation_ladder() -> None:
    """Each arm differs from the one before it by exactly one thing, or the table reads as noise."""
    assert list(VARIANTS) == ["zero_shot", "few_shot"]
    assert FEW_SHOT.startswith(ZERO_SHOT), "few_shot adds examples and nothing else"
    assert not any(v.cache for v in VARIANTS.values()), "the prompt ablation holds caching fixed"


def test_the_caching_arms_move_only_the_breakpoint() -> None:
    """The cache is a serving detail, not a different request. If the two arms differ by so much
    as a character of prompt, a score gap between them is unattributable."""
    systems = {v.system for v in CACHE_VARIANTS.values()}
    assert len(systems) == 1, "one prompt across both arms"
    assert [v.cache for v in CACHE_VARIANTS.values()] == [False, True], "off, then on"


@pytest.mark.parametrize("cache", [False, True], ids=["plain", "breakpoint"])
def test_a_variant_sends_the_block_form_only_when_it_caches(cache: bool) -> None:
    sent = Variant(ZERO_SHOT, cache=cache).as_sent()

    if not cache:
        assert sent == ZERO_SHOT, "a plain prompt goes on the wire as a plain string"
        return
    assert sent == [{"type": "text", "text": ZERO_SHOT, "cache_control": {"type": "ephemeral"}}], (
        "a breakpoint needs the block form, and it goes on the last system block"
    )


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


@pytest.mark.parametrize("ticker", sorted(DIRECTORY), ids=lambda t: t)
def test_every_spelling_the_directory_lists_resolves_to_its_own_ticker(ticker: str) -> None:
    resolver = Resolver()
    for spelling in forms(ticker):
        assert resolver(spelling) == ticker, f"{spelling!r} should resolve to {ticker}"


@pytest.mark.parametrize("ticker", sorted(DIRECTORY), ids=lambda t: t)
def test_no_company_lists_the_same_spelling_twice(ticker: str) -> None:
    """Matching is case-insensitive, so "PVR INOX" and "PVR Inox" are one spelling written twice,
    and an alias restating the ticker or the registered name is a third copy of something already
    indexed. Harmless to the lookup; noise in a file that reads as though every line did work."""
    folded = [" ".join(spelling.split()).casefold() for spelling in forms(ticker)]
    assert len(set(folded)) == len(folded), f"{ticker}: duplicates in {forms(ticker)}"


def test_no_two_companies_claim_the_same_spelling() -> None:
    """`Resolver` indexes with `setdefault`, so a collision would hand every such mention to
    whichever company happened to be first. SBI and SBI Cards are the near miss this guards."""
    owners: dict[str, list[str]] = {}
    for ticker in DIRECTORY:
        for spelling in forms(ticker):
            owners.setdefault(_key(spelling), []).append(ticker)
    clashes = {key: sorted(set(t)) for key, t in owners.items() if len(set(t)) > 1}
    assert not clashes, f"one spelling, two companies: {clashes}"


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


def test_a_headline_with_no_single_company_never_reaches_the_directory() -> None:
    """`null` is an answer, not a miss. A sector headline has no company to look up, so the
    lookup must not run and must not file a to-do against a name nobody wrote."""
    resolver = Resolver()
    task, _ = _task_and_log(
        record=Extraction(
            company=None, metric="none", quarter="Q3", direction="unknown", change_pct=None
        ),
        resolver=resolver,
    )
    outcome = task(_case("Q3 results: Nifty IT companies post mixed numbers"))

    assert outcome.error is None
    assert outcome.output["ticker"] is None and outcome.output["company"] is None
    assert not resolver.unresolved, "nothing to add to the directory; there was no company"


def test_the_directory_covers_every_ticker_the_dataset_labels() -> None:
    """The directory is the ground truth for `ticker`. A label outside it is unfalsifiable."""
    labelled = {str(case.expected["ticker"]) for case in NAMED}
    assert labelled <= set(DIRECTORY), f"not in the directory: {sorted(labelled - set(DIRECTORY))}"


@pytest.mark.parametrize("case", NAMED, ids=lambda case: case.id)
def test_every_headline_names_its_company_in_a_form_the_directory_lists(case: Case) -> None:
    """The spelling the headline uses has to be resolvable, or the label is a leap the model
    cannot make and the directory cannot help with."""
    headline = str(case.input["headline"]).casefold()
    ticker = str(case.expected["ticker"])
    assert any(spelling.casefold() in headline for spelling in forms(ticker)), (
        f"{case.id}: none of {forms(ticker)} appears in {headline!r}"
    )


@pytest.mark.parametrize("case", NAMED, ids=lambda case: case.id)
def test_copying_the_headline_correctly_is_enough_to_get_the_ticker_right(case: Case) -> None:
    """The property the whole design rests on: the model only has to copy, and the ticker follows.

    For each case, take the spelling the headline actually uses — what a careful model would
    return — and check the resolver lands on the label. While this holds for every row, `ticker` is
    right by construction whenever `company` is, and the eval measures identification, not recall.
    """
    expected = str(case.expected["ticker"])
    assert Resolver()(written_as(case)) == expected


@pytest.mark.parametrize("case", NAMED, ids=lambda case: case.id)
def test_a_company_whose_name_hides_another_is_tagged(case: Case) -> None:
    """`Tech Mahindra` contains `Mahindra`, which is M&M. A model that truncates the name resolves
    to a real but wrong company, and the resolver reports success — the one silent failure this
    design has. The tag marks those rows so the report can be sliced by them."""
    ticker = str(case.expected["ticker"])
    shadowed = shadowed_by(written_as(case), ticker)
    tagged = "name-contains-name" in case.tags

    assert tagged == bool(shadowed), (
        f"{case.id}: {written_as(case)!r} hides {shadowed}, tag it name-contains-name"
        if shadowed
        else f"{case.id}: nothing hides inside {written_as(case)!r}, drop the tag"
    )


def test_the_metric_order_is_total_and_covers_the_vocabulary() -> None:
    """A headline routinely names two metrics. Without a total order the label is a coin flip the
    model has no way to call, and the miss reads as a model failure when it is a spec failure."""
    assert set(METRICS) == VOCABULARY["metric"], "every metric is ranked, and nothing extra is"
    assert len(set(METRICS)) == len(METRICS), "a total order has no ties"

    description = Extraction.model_fields["metric"].description or ""
    assert ", ".join(METRICS) in description, (
        "the model is shown the order, not just told there is one"
    )
    # Without this, "cuts revenue guidance" cues revenue, which outranks guidance.
    assert "forward-looking" in description and "guidance, not that metric" in description


def test_both_variants_state_the_conventions_the_dataset_is_labelled_by() -> None:
    """A rule that only appears in the few-shot examples would punish zero_shot for not guessing."""
    descriptions = " ".join(field.description or "" for field in Extraction.model_fields.values())
    for convention in (
        "verbatim",
        "quarters ending June",
        "basis",
        "unknown",
        "listed parent",
        "no single company",
        "share price move",
        "rate of growth",
        "bare month",
    ):
        assert convention in descriptions, convention


def test_no_illustration_in_the_prompt_is_a_row_of_the_dataset() -> None:
    """A worked example that is also a test case stops that row measuring anything.

    Four words is the window: it catches an example built from a row — three words fires on
    financial boilerplate like "Q1 revenue up", which two headlines can share innocently.
    """
    prompt = FEW_SHOT + " ".join(f.description or "" for f in Extraction.model_fields.values())
    quoted = word_runs(prompt)
    shared = {
        case.id: sorted(" ".join(g) for g in word_runs(str(case.input["headline"])) & quoted)
        for case in LABELLED
    }
    leaked = {case_id: overlap for case_id, overlap in shared.items() if overlap}
    assert not leaked, f"the prompt quotes the dataset: {leaked}"


# --- the dataset is an asset, so it gets tested like one ----------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4, 7, len(LABELLED), len(LABELLED) + 50])
def test_a_sample_spans_the_dataset_rather_than_taking_the_front(n: int) -> None:
    """The rows are roughly in the order they were written, so the front is the easy end and a
    smoke run taken off it cannot fail.

    Asserted as properties, not as a list of ids: the ids change every time the dataset grows, and
    a test that breaks for that reason is noise.
    """
    at = {case.id: i for i, case in enumerate(LABELLED)}
    picked = [at[case.id] for case in spread(LABELLED, n)]

    assert len(picked) == min(n, len(LABELLED))
    assert picked == sorted(picked), "order is preserved"
    assert picked[0] == 0, "the sample starts at the top"
    if 1 < n < len(LABELLED):  # a sample of one can only be the first row
        gaps = {b - a for a, b in pairwise(picked)}
        assert len(gaps) <= 1, "evenly spaced"
        assert picked[-1] >= n, "and it reaches past the first n, or there was no point"


def test_the_dataset_is_the_size_the_project_asked_for() -> None:
    assert 40 <= len(CASES) <= 60, "Project 1a: 40 to 60 inputs"
    assert len(LABELLED) >= 30, "30 of them hand-labelled"


@pytest.mark.parametrize("case", LABELLED, ids=lambda case: case.id)
def test_every_label_validates_against_the_vocabulary(case: Case) -> None:
    # The model never produces `ticker`, so fill `company` in from the directory to validate.
    ticker = case.expected["ticker"]
    name = DIRECTORY[str(ticker)]["name"] if ticker is not None else None
    record = Extraction.model_validate({**case.expected, "company": name})

    assert set(case.expected) == set(FIELDS), "a label with a missing field grades as wrong forever"
    assert ticker is None or TICKER.fullmatch(str(ticker)), f"{case.id}: {ticker!r} is not a ticker"
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
