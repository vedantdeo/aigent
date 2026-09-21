"""The one module that talks to a model: counted, admitted, sent, billed, traced — every call shape.

The load-bearing test is the batch one. Calls in flight cannot be recalled, so a ceiling checked
call by call is a ceiling that concurrent calls walk straight past. The last four are the failure
catalogue that used to live in `primitives.failures`: each way a call can go wrong, and where
`llm` stops it — always before anything billable is sent.
"""

from __future__ import annotations

import ast
import time
from dataclasses import fields
from pathlib import Path

import anthropic
import httpx2
import pytest
from anthropic.types import Message, ToolParam, ToolUseBlock
from pydantic import BaseModel, ValidationError

import entropic
from entropic.config import MODEL
from entropic.llm import Rehearsed, Request, StepFailed, describe
from entropic.pricing import BudgetExceeded, worst_case_usd

from .conftest import FAKE_USAGE, MakeLlm, Sent

# Worst case on the fake: 100 input tokens at $5/M plus 64 output at $25/M, $0.0021.
REQUEST = Request.ask("greet", "be brief", "hello", 64)
LOOKUP: ToolParam = {
    "name": "lookup",
    "description": "look it up",
    "input_schema": {"type": "object"},
}


class Greeting(BaseModel):
    words: str


def echo(sent: Sent) -> str:
    return f"re: {sent.prompt}"


def batch(n: int) -> list[Request]:
    return [Request.ask(f"part:{i}", "be brief", str(i), 64) for i in range(n)]


def test_a_call_is_counted_then_sent_then_billed_and_traced(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(echo)

    assert llm.text(REQUEST) == "re: hello"

    assert fake.counted == [None] and len(fake.sent) == 1
    [call] = llm.trace
    assert (call.step, call.model) == ("greet", MODEL)
    assert call.usd > 0 and llm.spent_usd == pytest.approx(call.usd)


def test_the_request_reaches_the_wire_as_written(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(echo)
    request = Request(
        "turn",
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"}],
        64,
        system="a system",
        model="claude-sonnet-5",
        tools=[LOOKUP],
        thinking={"type": "disabled"},
    )

    llm.text(request)

    [sent] = fake.sent
    assert (sent.model, sent.system, sent.messages) == (
        "claude-sonnet-5",
        "a system",
        request.messages,
    )
    assert (sent.tools, sent.thinking) == ([LOOKUP], {"type": "disabled"})


def test_counting_is_free(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(echo)

    assert llm.count(REQUEST, Greeting) == FAKE_USAGE.input_tokens
    assert fake.sent == [] and llm.trace == [] and llm.spent_usd == 0.0
    assert fake.counted == [Greeting], "the schema is input tokens too, so it is counted"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_text_refuses_an_unusable_reply_but_bills_it(make_llm: MakeLlm, stop_reason: str) -> None:
    llm, _ = make_llm(echo, stop_reason=stop_reason)

    with pytest.raises(StepFailed, match=stop_reason):
        llm.text(REQUEST)
    assert len(llm.trace) == 1, "the call happened, so it is billed"


def test_create_hands_back_the_whole_message_for_the_caller_to_read(make_llm: MakeLlm) -> None:
    """A tool loop reads `stop_reason` and the `tool_use` blocks itself, so `create` judges
    nothing: a turn that stops to use a tool is a turn, not a failure."""
    wants_tool = Message(
        id="msg_tool",
        type="message",
        role="assistant",
        model=MODEL,
        content=[ToolUseBlock(type="tool_use", id="toolu_1", name="lookup", input={})],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=FAKE_USAGE,
    )
    llm, _ = make_llm(lambda sent: wants_tool)

    assert llm.create(REQUEST) is wants_tool
    assert len(llm.trace) == 1


def test_parse_reports_nothing_parsed_where_record_refuses_it(make_llm: MakeLlm) -> None:
    """Two readings of one outcome: an eval task wants the message and records the miss as a row;
    a workflow step wants the record and cannot go on without it. Both calls are billed."""
    llm, fake = make_llm(lambda sent: None)

    assert llm.parse(REQUEST, Greeting).parsed_output is None
    with pytest.raises(StepFailed, match="nothing parsed"):
        llm.record(REQUEST, Greeting)
    assert len(llm.trace) == 2
    assert fake.counted == [Greeting, Greeting]


def _cut_off() -> ValidationError:
    try:
        Greeting.model_validate_json('{"words": "cut o')
    except ValidationError as error:
        return error
    raise AssertionError("truncated JSON should not validate")


def test_a_reply_the_sdk_cannot_read_is_billed_at_its_worst_case(make_llm: MakeLlm) -> None:
    """The SDK validates a structured reply as it reads it, so a record cut off at `max_tokens`
    raises before its usage is returned — and a call raised past is a call nobody billed. It comes
    back as nothing parsed, billed at the worst case, which is what a cut-off reply costs."""
    llm, _ = make_llm(lambda sent: _cut_off())

    response = llm.parse(REQUEST, Greeting)

    assert response.parsed_output is None and response.stop_reason == "max_tokens"
    [call] = llm.trace
    assert call.usage.output_tokens == REQUEST.max_tokens
    assert call.usd == pytest.approx(worst_case_usd(MODEL, FAKE_USAGE.input_tokens, 64))
    with pytest.raises(StepFailed, match="nothing parsed"):
        llm.record(REQUEST, Greeting)
    assert len(llm.trace) == 2, "the refused record is billed too"


def test_record_returns_the_validated_record(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(lambda sent: Greeting(words="hi"))

    assert llm.record(REQUEST, Greeting) == Greeting(words="hi")
    assert fake.sent[0].schema is Greeting


def test_a_call_that_could_cross_the_ceiling_is_refused_unsent(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(echo, limit_usd=0.002)

    with pytest.raises(BudgetExceeded, match="per-run ceiling"):
        llm.text(REQUEST)
    assert fake.sent == []


def test_a_batch_is_admitted_whole_or_not_at_all(make_llm: MakeLlm) -> None:
    """Two of these fit under the ceiling and three do not, so none may go: admitting them one at
    a time would send the first two and find the problem with the third already paid for."""
    llm, fake = make_llm(echo, limit_usd=0.005)

    with pytest.raises(BudgetExceeded):
        llm.gather_text(batch(3))
    assert fake.sent == [], "not one call goes out when the whole batch cannot"


def test_a_batch_answers_in_request_order_whatever_order_it_finishes(make_llm: MakeLlm) -> None:
    def first_is_slowest(sent: Sent) -> str:
        if sent.prompt == "0":
            time.sleep(0.05)
        return sent.prompt

    llm, fake = make_llm(first_is_slowest)

    assert llm.gather_text(batch(3)) == ["0", "1", "2"]
    assert len(fake.counted) == 3, "counted once each, to admit the whole"


def test_a_batch_of_records_is_priced_with_its_schema(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(lambda sent: Greeting(words=sent.prompt))

    greetings = llm.gather_records(batch(2), Greeting)

    assert [greeting.words for greeting in greetings] == ["0", "1"]
    assert fake.counted == [Greeting, Greeting]


def test_a_stream_is_admitted_first_and_billed_once_it_closes(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(lambda sent: "streamed reply")

    with llm.stream(REQUEST) as stream:
        streamed = " ".join(stream.text_stream)
        assert llm.trace == [], "nothing is billed until the final message exists"

    assert streamed == "streamed reply"
    assert [sent.kind for sent in fake.sent] == ["stream"] and fake.counted == [None]
    assert len(llm.trace) == 1


def test_a_rehearsal_counts_the_first_call_and_sends_nothing(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(echo, rehearse=True)

    with pytest.raises(Rehearsed) as caught:
        llm.text(REQUEST)

    assert caught.value.input_tokens == FAKE_USAGE.input_tokens
    assert caught.value.worst_usd == pytest.approx(worst_case_usd(MODEL, 100, 64))
    assert fake.sent == [] and llm.trace == []


def test_the_trace_prints_a_line_per_call_and_the_total(make_llm: MakeLlm) -> None:
    llm, _ = make_llm(echo)
    llm.text(REQUEST)
    llm.text(Request.ask("again", "be brief", "x", 64))

    lines = describe(llm.trace).splitlines()

    assert [line.split()[0] for line in lines[1:3]] == ["greet", "again"]
    assert lines[-1].split() == ["2", "calls", f"{llm.spent_usd:.5f}"]


def _rejected(error: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens")
    return error("rejected", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        pytest.param(anthropic.NotFoundError, 404, id="a model that does not exist"),
        pytest.param(anthropic.AuthenticationError, 401, id="credentials that are not ours"),
        pytest.param(anthropic.BadRequestError, 400, id="a request with nothing to answer"),
    ],
)
def test_what_only_the_api_can_reject_is_rejected_at_the_free_count(
    make_llm: MakeLlm, error: type[anthropic.APIStatusError], status: int
) -> None:
    """Every call starts with the count, which is free, so a request the API refuses is refused
    there — before anything that could be billed is sent."""
    llm, fake = make_llm(echo)
    fake.count_error = _rejected(error, status)

    with pytest.raises(error):
        llm.text(REQUEST)
    assert fake.sent == [] and llm.trace == []


@pytest.mark.parametrize(
    ("call", "input_tokens", "spent", "says"),
    [
        pytest.param(
            Request.ask("x", "be brief", "hi", 64, model="claude-opus-4-8"),
            100,
            0.0,
            "has no price",
            id="a real model with no price",
        ),
        pytest.param(
            Request.ask("x", "be brief", "hi", 10_000_000),
            100,
            0.0,
            "per-request ceiling",
            id="an output cap no single call may carry",
        ),
        pytest.param(REQUEST, 2_000_000, 0.0, "per-request ceiling", id="an input too large"),
        pytest.param(REQUEST, 100, 0.999, "per-run ceiling", id="a run with too little left"),
    ],
)
def test_what_our_own_guards_refuse_is_never_sent(
    make_llm: MakeLlm, call: Request, input_tokens: int, spent: float, says: str
) -> None:
    llm, fake = make_llm(echo)
    fake.input_tokens = input_tokens
    llm.budget.spent_usd = spent

    with pytest.raises(BudgetExceeded, match=says):
        llm.text(call)
    assert fake.sent == []


def test_a_request_has_no_field_for_what_the_api_rejects() -> None:
    """Claude 5 rejects `temperature` and `top_p`, and every model rejects a field it does not
    know. A `Request` has no way to carry either, so neither can be sent from here."""
    offered = {field.name for field in fields(Request)}
    assert offered.isdisjoint({"temperature", "top_p", "top_k", "extra_body"})


def _talks_to_the_model(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"create", "parse", "stream", "count_tokens"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "messages"
        ):
            return True
    return False


def test_only_llm_talks_to_the_model() -> None:
    """Every count, send and stream in the package goes through `llm`, so every one is counted,
    checked, admitted and billed. A module calling `client.messages` itself skips all four."""
    package = Path(entropic.__file__).parent
    talkers = {
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if _talks_to_the_model(path)
    }
    assert talkers == {"llm.py"}
