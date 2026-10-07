"""`Llm.batch_records` against the fake client's batch endpoint: admitted whole at the batch price,
waited for, billed at half price per result in request order, and every failure kept as data."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse
from pydantic import BaseModel

from aigent import llm as llm_module
from aigent.adapters.anthropic import Anthropic, batch_params, batch_result
from aigent.adapters.cfg_anthropic import CLIENT as ANTHROPIC
from aigent.config import BATCH_POLL_SECONDS, MODEL
from aigent.errors import BatchUnfinished, BudgetExceeded, Unsupported
from aigent.llm import Llm, Rehearsed, Request
from aigent.messages import BatchStatus, Failed, Parsed, Usage
from aigent.pricing import Budget, usage_cost, worst_case_usd

from .conftest import FAKE_USAGE, FakeAnthropic, MakeLlm, Recorder, Sent


class Answer(BaseModel):
    answer: str


def scripted(sent: Sent) -> Answer | str | Exception | None:
    """A reply named by the prompt: most are records, a few are each way a request can fail."""
    return {
        "fail": ValueError("prompt is too long"),
        "expire": None,
        "prose": "not a record at all",
    }.get(sent.prompt, Answer(answer=f"re: {sent.prompt}"))


def ask(prompt: str, max_tokens: int = 64) -> Request:
    return Request.ask("headline", "Answer.", prompt, max_tokens)


FULL = Usage(FAKE_USAGE.input_tokens, FAKE_USAGE.output_tokens)
BATCHED = Usage(FULL.input_tokens, FULL.output_tokens, batched=True)


@pytest.fixture
def waited(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Every pause the poll loop takes, recorded instead of slept."""
    pauses: list[float] = []
    monkeypatch.setattr(llm_module, "_sleep", pauses.append)
    yield pauses


def test_results_come_back_in_request_order_billed_at_half_price(
    make_llm: MakeLlm, waited: list[float]
) -> None:
    llm, fake = make_llm(scripted)
    prompts = ["a", "b", "c"]

    results = llm.batch_records([ask(prompt) for prompt in prompts], Answer)

    records = [result.parsed for result in results if isinstance(result, Parsed)]
    assert records == [Answer(answer=f"re: {prompt}") for prompt in prompts], "reversed by the fake"
    assert [call.usage for call in llm.trace] == [BATCHED] * 3
    assert llm.spent_usd == pytest.approx(3 * usage_cost(MODEL, BATCHED))
    assert llm.spent_usd == pytest.approx(3 * usage_cost(MODEL, FULL) / 2)
    assert llm.budget.held_usd == 0, "the hold is let go once every result is billed"
    assert len(fake.batches.created) == 1 and fake.sent == [
        sent for sent in fake.sent if sent.kind == "batch"
    ], "one batch, no single calls"


def test_each_request_is_sent_under_its_position_with_its_schema(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(scripted)

    llm.batch_records([ask("a"), ask("b")], Answer)

    [submitted] = fake.batches.created
    assert [request["custom_id"] for request in submitted] == ["r0", "r1"]
    params = cast(dict[str, dict[str, dict[str, object]]], submitted[0]["params"])
    assert params["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize(
    ("prompt", "reason"),
    [
        pytest.param("fail", "errored: invalid_request_error: prompt is too long", id="an error"),
        pytest.param("expire", "expired", id="a request the 24 hours ran out on"),
    ],
)
def test_a_request_with_no_reply_comes_back_failed_and_unbilled(
    prompt: str, reason: str, make_llm: MakeLlm, traces: Recorder
) -> None:
    llm, _ = make_llm(scripted)

    ok, failed = llm.batch_records([ask("a"), ask(prompt)], Answer)

    assert isinstance(ok, Parsed) and failed == Failed(reason)
    assert len(llm.trace) == 1, "only the request that was answered is billed"
    generations = [node for node in traces.every() if node.kind == "generation"]
    assert [node.error for node in generations] == [None, reason]
    assert generations[1].billed is None


def test_a_reply_that_is_not_a_record_is_billed_with_nothing_parsed(make_llm: MakeLlm) -> None:
    llm, _ = make_llm(scripted)

    [result] = llm.batch_records([ask("prose")], Answer)

    assert isinstance(result, Parsed) and result.parsed is None
    assert result.text == "not a record at all"
    assert llm.spent_usd == pytest.approx(usage_cost(MODEL, BATCHED))


def test_a_batch_is_admitted_whole_at_the_batch_price() -> None:
    fake = FakeAnthropic(scripted)
    one = worst_case_usd(MODEL, FAKE_USAGE.input_tokens, 4096)
    llm = Llm(sdk=fake, budget=Budget(limit_usd=3 * one / 2 * 1.01))

    llm.batch_records([ask(prompt, 4096) for prompt in "abc"], Answer)

    assert len(fake.messages.batches.created) == 1, "three fit at half price, not at full"
    with pytest.raises(BudgetExceeded):
        llm.batch_records([ask(prompt, 4096) for prompt in "abc"], Answer)
    assert len(fake.messages.batches.created) == 1, "a batch that cannot all fit is not sent"


def test_a_dry_run_prices_the_first_request_at_the_batch_price(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(scripted, rehearse=True)

    with pytest.raises(Rehearsed) as caught:
        llm.batch_records([ask("a", 4096)], Answer)

    full = worst_case_usd(MODEL, FAKE_USAGE.input_tokens, 4096)
    assert caught.value.worst_usd == pytest.approx(full / 2)
    assert fake.batches.created == []


def test_it_polls_until_the_batch_ends(make_llm: MakeLlm, waited: list[float]) -> None:
    llm, fake = make_llm(scripted)
    fake.batches.polls = 3

    llm.batch_records([ask("a")], Answer)

    assert fake.batches.checked == 4
    assert waited == [BATCH_POLL_SECONDS] * 3


@pytest.fixture
def stalled(make_llm: MakeLlm, monkeypatch: pytest.MonkeyPatch, waited: list[float]) -> Llm:
    """An `Llm` whose batch never ends, on a clock that moves one poll per look."""
    llm, fake = make_llm(scripted)
    fake.batches.polls = 10**6
    clock = iter(range(0, 10**6, BATCH_POLL_SECONDS))
    monkeypatch.setattr(llm_module, "_now", lambda: float(next(clock)))
    return llm


def test_a_batch_that_outlives_the_wait_is_named_so_it_can_be_collected(stalled: Llm) -> None:
    with pytest.raises(BatchUnfinished, match="msgbatch_1") as caught:
        stalled.collect_batch("msgbatch_1", [ask("a")], Answer, wait_seconds=90)

    assert "still processing" in str(caught.value)


def test_a_batch_that_outlives_the_wait_keeps_its_hold(stalled: Llm) -> None:
    with pytest.raises(BatchUnfinished):
        stalled.batch_records([ask("a")], Answer)

    assert stalled.budget.held_usd > 0, "its requests may still be billed, so the hold stays"
    assert stalled.spent_usd == 0


def test_a_batch_whose_results_cannot_be_read_keeps_its_hold(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(scripted)
    fake.batches.results_error = ConnectionError("reset by peer")

    with pytest.raises(BatchUnfinished, match="reset by peer"):
        llm.batch_records([ask("a")], Answer)

    assert llm.budget.held_usd > 0


def test_a_batch_that_could_not_be_submitted_lets_its_hold_go(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(scripted)
    fake.batches.submit_error = ConnectionError("refused")

    with pytest.raises(ConnectionError):
        llm.batch_records([ask("a")], Answer)

    assert llm.budget.held_usd == 0


def test_a_batch_collected_later_bills_what_came_back() -> None:
    fake = FakeAnthropic(scripted)
    requests = [ask("a"), ask("fail")]
    fake.messages.batches.results_error = ConnectionError("lost")
    with pytest.raises(BatchUnfinished):
        Llm(sdk=fake, budget=Budget(limit_usd=1.0)).batch_records(requests, Answer)
    fake.messages.batches.results_error = None

    later = Llm(sdk=fake, budget=Budget(limit_usd=1.0))
    ok, failed = later.collect_batch("msgbatch_1", requests, Answer)

    assert isinstance(ok, Parsed) and isinstance(failed, Failed)
    assert later.spent_usd == pytest.approx(usage_cost(MODEL, BATCHED))
    assert later.budget.held_usd == 0, "a batch collected later was never held here"


def test_a_result_missing_from_the_batch_comes_back_failed(
    make_llm: MakeLlm, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm, fake = make_llm(scripted)
    monkeypatch.setattr(fake.batches, "results", lambda batch_id: [])

    [result] = llm.batch_records([ask("a")], Answer)

    assert result == Failed("no result came back") and llm.spent_usd == 0


def test_a_wire_with_no_batches_is_refused_before_anything_is_counted() -> None:
    llm = Llm("local-1.7b-toy")

    with pytest.raises(Unsupported, match="does not support batch"):
        llm.batch_records([ask("a")], Answer)


@pytest.mark.parametrize(
    ("request_", "expected"),
    [
        pytest.param(
            ask("a"),
            {"system": "Answer.", "output_config": {"format": "json_schema"}},
            id="a plain request carries its system prompt and the schema",
        ),
        pytest.param(
            Request(
                "headline",
                [{"role": "user", "content": "a"}],
                64,
                thinking={"type": "disabled"},
                output_config={"effort": "low"},
                cache_control={"type": "ephemeral"},
            ),
            {
                "thinking": {"type": "disabled"},
                "cache_control": {"type": "ephemeral"},
                "output_config": {"effort": "low", "format": "json_schema"},
            },
            id="effort survives beside the schema, and thinking and caching are sent",
        ),
    ],
)
def test_a_request_is_carried_in_a_batch_as_it_would_be_sent(
    request_: Request, expected: dict[str, object]
) -> None:
    params = batch_params(request_, ANTHROPIC, Answer)

    output_config = cast(dict[str, object], params.pop("output_config"))
    shown = {**output_config, "format": cast(dict[str, object], output_config["format"])["type"]}
    assert {**params, "output_config": shown} == {
        "model": MODEL,
        "max_tokens": 64,
        "messages": list(request_.messages),
        **expected,
    }


def test_a_canceled_request_comes_back_failed() -> None:
    response = MessageBatchIndividualResponse.model_validate(
        {"custom_id": "r0", "result": {"type": "canceled"}}
    )

    assert batch_result(response, Answer) == Failed("canceled")


def test_progress_counts_every_way_a_request_can_fail_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAnthropic(scripted)
    counts = {"processing": 1, "succeeded": 2, "errored": 3, "expired": 4, "canceled": 5}
    batch = MessageBatch.model_validate(
        {
            "id": "msgbatch_1",
            "type": "message_batch",
            "processing_status": "in_progress",
            "request_counts": counts,
            "created_at": "2026-10-07T00:00:00Z",
            "expires_at": "2026-10-08T00:00:00Z",
        }
    )
    monkeypatch.setattr(fake.messages.batches, "retrieve", lambda batch_id: batch)

    status = Anthropic(ANTHROPIC, fake).poll("msgbatch_1")

    assert status == BatchStatus(ended=False, processing=1, succeeded=2, failed=12)
