"""The one module that talks to a model: counted, admitted, sent, billed, traced — every call shape.

The load-bearing test is the batch one. Calls in flight cannot be recalled, so a ceiling checked
call by call is a ceiling that concurrent calls walk straight past. The last four are the failure
catalogue that used to live in `primitives.failures`: each way a call can go wrong, and where
`llm` stops it — always before anything billable is sent.
"""

from __future__ import annotations

import ast
import time
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import cast

import anthropic
import httpx2
import pytest
from anthropic.types import (
    Message,
    ServerToolUsage,
    TextBlock,
    ToolParam,
    ToolUseBlock,
    Usage,
)
from pydantic import BaseModel, ValidationError

import entropic
from entropic.config import MAX_USD_PER_TURN, MODEL
from entropic.errors import BudgetExceeded, StepFailed, TurnsExhausted
from entropic.llm import Dispatch, Llm, Rehearsed, Request, describe
from entropic.pricing import PRICES, affordable_output_tokens, worst_case_usd
from entropic.tools import ALL_TOOLS, WEB_SEARCH_TOOL, execute_tool
from entropic.tools_config import MAX_WEB_SEARCHES

from .conftest import FAKE_USAGE, FakeMessages, MakeLlm, Sent, tool_results, tool_turn, turns

# Worst case on the fake: 100 input tokens at $5/M plus 64 output at $25/M, $0.0021.
REQUEST = Request.ask("greet", "be brief", "hello", 64)
CACHED = replace(REQUEST, cache_control={"type": "ephemeral"})
MARKED = replace(
    REQUEST, system=[{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}]
)
SEARCHING_WEB = replace(REQUEST, tools=[WEB_SEARCH_TOOL])
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


@pytest.mark.parametrize(
    ("asked", "cached"),
    [
        pytest.param(REQUEST, False, id="an uncached call"),
        pytest.param(CACHED, True, id="a cached call, billed as a full cache write"),
    ],
)
def test_a_reply_the_sdk_cannot_read_is_billed_at_its_worst_case(
    make_llm: MakeLlm, asked: Request, cached: bool
) -> None:
    """The SDK validates a structured reply as it reads it, so a record cut off at `max_tokens`
    raises before its usage is returned — and a call raised past is a call nobody billed. It comes
    back as nothing parsed, billed at the worst case, which is what a cut-off reply costs."""
    llm, _ = make_llm(lambda sent: _cut_off())

    response = llm.parse(asked, Greeting)

    assert response.parsed_output is None and response.stop_reason == "max_tokens"
    [call] = llm.trace
    assert call.usage.output_tokens == asked.max_tokens
    worst = worst_case_usd(MODEL, FAKE_USAGE.input_tokens, 64, cached=cached)
    assert call.usd == pytest.approx(worst), call.usage
    with pytest.raises(StepFailed, match="nothing parsed"):
        llm.record(asked, Greeting)
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


def test_a_call_holds_its_worst_case_while_in_flight_and_lets_go_once_billed(
    make_llm: MakeLlm,
) -> None:
    seen: list[float] = []
    holder: list[Llm] = []

    def reply(sent: Sent) -> str:
        seen.append(holder[0].budget.held_usd)
        return "hi"

    llm, _ = make_llm(reply)
    holder.append(llm)

    llm.text(REQUEST)

    assert seen == [pytest.approx(worst_case_usd(MODEL, FAKE_USAGE.input_tokens, 64))]
    assert llm.budget.held_usd == 0.0


def test_a_failed_call_lets_go_of_its_hold(make_llm: MakeLlm) -> None:
    llm, _ = make_llm(lambda sent: RuntimeError("connection reset"))

    with pytest.raises(RuntimeError):
        llm.text(REQUEST)

    assert llm.budget.held_usd == 0.0, "a hold that outlives its call shrinks every later ceiling"


def test_a_call_admitted_while_another_is_in_flight_counts_it(make_llm: MakeLlm) -> None:
    """What a framework's parallel nodes do: two calls, each admitted alone, both in flight. One
    fits under this ceiling and two do not, so the second is refused while the first runs."""
    refused: list[str] = []
    holder: list[Llm] = []

    def reply(sent: Sent) -> str:
        if sent.prompt == "hello":
            try:
                holder[0].text(Request.ask("second", "be brief", "again", 64))
            except BudgetExceeded as error:
                refused.append(str(error))
        return "hi"

    llm, fake = make_llm(reply, limit_usd=0.004)
    holder.append(llm)

    llm.text(REQUEST)

    assert len(refused) == 1 and "held for calls in flight" in refused[0], refused
    assert [sent.prompt for sent in fake.sent] == ["hello"], "the second call was never sent"


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
    assert llm.budget.held_usd == 0.0, "the batch's hold is let go once it is billed"


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


@pytest.mark.parametrize(
    ("asked", "cached", "searches"),
    [
        pytest.param(REQUEST, False, 0, id="an uncached call pays the input price at worst"),
        pytest.param(CACHED, True, 0, id="automatic caching can write every input token"),
        pytest.param(MARKED, True, 0, id="so can a marked system block"),
        pytest.param(
            SEARCHING_WEB, False, MAX_WEB_SEARCHES, id="a capped web search can run to its cap"
        ),
    ],
)
def test_a_call_is_admitted_at_the_most_it_could_cost(
    make_llm: MakeLlm, asked: Request, cached: bool, searches: int
) -> None:
    """A cache miss writes the whole prefix at 1.25x, and a capped search can run to its cap."""
    llm, _ = make_llm(echo, rehearse=True)

    with pytest.raises(Rehearsed) as caught:
        llm.text(asked)

    worst = worst_case_usd(MODEL, FAKE_USAGE.input_tokens, 64, cached=cached, web_searches=searches)
    assert caught.value.worst_usd == pytest.approx(worst), asked.system


def _streamed(llm: Llm) -> str:
    with llm.stream(CACHED) as stream:
        return "".join(stream.text_stream)


def _looped(llm: Llm) -> object:
    return llm.run_tools(replace(CACHED, tools=[LOOKUP]), lambda name, args: ("found", False))


@pytest.mark.parametrize(
    "send",
    [
        pytest.param(lambda llm: llm.text(CACHED), id="create"),
        pytest.param(lambda llm: llm.parse(CACHED, Greeting), id="parse, through the body"),
        pytest.param(_streamed, id="stream"),
        pytest.param(_looped, id="the tool runner"),
    ],
)
def test_caching_reaches_the_wire_on_every_path(
    make_llm: MakeLlm, send: Callable[[Llm], object]
) -> None:
    llm, fake = make_llm(lambda sent: Greeting(words="hi") if sent.kind == "parse" else "hi")

    send(llm)

    [sent] = fake.sent
    assert sent.cache_control == {"type": "ephemeral"}, sent.kind


def test_the_trace_counts_cached_input_as_input_and_shows_web_searches(make_llm: MakeLlm) -> None:
    usage = Usage(
        input_tokens=10,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=70,
        output_tokens=5,
        server_tool_use=ServerToolUsage(web_search_requests=2, web_fetch_requests=0),
    )
    reply = Message(
        id="m",
        type="message",
        role="assistant",
        model=MODEL,
        content=[TextBlock(type="text", text="hi")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=usage,
    )
    llm, _ = make_llm(lambda sent: reply)
    llm.text(CACHED)

    [header, line, _] = describe(llm.trace).splitlines()

    assert header.split()[2:6] == ["in", "cached", "out", "web"]
    assert line.split()[2:6] == ["100", "70", "5", "2"], line


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
        pytest.param(
            replace(REQUEST, tools=[{"type": "web_search_20260318", "name": "web_search"}]),
            100,
            0.0,
            "no max_uses",
            id="a web search with no cap on how often it runs",
        ),
        pytest.param(
            replace(REQUEST, tools=[{"type": "web_fetch_20260209", "name": "web_fetch"}]),
            100,
            0.0,
            "pricing does not model",
            id="a server tool pricing does not model",
        ),
        pytest.param(
            replace(REQUEST, cache_control={"type": "ephemeral", "ttl": "1h"}),
            100,
            0.0,
            "1-hour cache",
            id="the 1-hour cache, whose writes are not priced",
        ),
        pytest.param(
            replace(
                REQUEST,
                system=[
                    {
                        "type": "text",
                        "text": "be brief",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            ),
            100,
            0.0,
            "1-hour cache",
            id="the 1-hour cache on a system block",
        ),
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
            and node.func.attr in {"create", "parse", "stream", "count_tokens", "tool_runner"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "messages"
        ):
            return True
    return False


def test_only_llm_talks_to_the_model() -> None:
    """Every count, send, stream and tool loop in the package goes through `llm`, so every one is
    counted, checked, admitted and billed. A module calling the SDK itself skips all four."""
    package = Path(entropic.__file__).parent
    talkers = {
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if _talks_to_the_model(path)
    }
    assert talkers == {"llm.py"}


AGENT = Request(
    "agent",
    [{"role": "user", "content": "what is 2 ** 10, and what time is it?"}],
    256,
    system="use the tools",
    tools=ALL_TOOLS,
)


def test_run_tools_runs_each_call_and_sends_every_result_in_one_message(make_llm: MakeLlm) -> None:
    """The runner's loop, driven from `llm`: the model asks for two tools in one turn, both run,
    both results travel back in a single user message (invariant 4), and each turn is counted,
    admitted and billed like any other call."""
    both = tool_turn(("t1", "calculate", {"expression": "2 ** 10"}), ("t2", "current_time", {}))
    llm, fake = make_llm(turns(both, "1024, and it is noon"))

    ran = llm.run_tools(AGENT, execute_tool)

    assert ran.cut_short is None and llm.budget.held_usd == 0.0
    assert len(ran.turns) == 2 and ran.turns[-1] is ran.message, "every turn, the last included"
    assert [block.text for block in ran.message.content if block.type == "text"] == [
        "1024, and it is noon"
    ]
    results = tool_results(fake.sent[1])
    assert [result["tool_use_id"] for result in results] == ["t1", "t2"]
    assert results[0]["content"] == "1024.0"
    assert [call.step for call in llm.trace] == ["agent:1", "agent:2"]
    assert len(fake.counted) == 2, "each turn counted once, to admit it"


def test_a_tool_error_goes_back_to_the_model_as_an_error_result(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(turns(tool_turn(("t1", "calculate", {"expression": "1 / 0"})), "sorry"))

    llm.run_tools(AGENT, execute_tool)

    [result] = tool_results(fake.sent[1])
    assert result.get("is_error") is True
    assert str(result["content"]).startswith("Error:"), result


def test_a_turn_that_cannot_be_afforded_is_never_sent(make_llm: MakeLlm) -> None:
    """A turn's worst case here is $0.0069 and the first costs $0.00175, so a $0.008 ceiling admits
    the first and not the second — which the runner would otherwise send without asking."""
    llm, fake = make_llm(turns(tool_turn(("t1", "current_time", {})), "done"), limit_usd=0.008)

    with pytest.raises(BudgetExceeded, match="per-run ceiling"):
        llm.run_tools(AGENT, execute_tool)
    assert [sent.kind for sent in fake.sent] == ["turn"]
    assert llm.budget.held_usd == 0.0, "a refused turn holds nothing"


def test_running_out_of_turns_raises_rather_than_looking_finished(make_llm: MakeLlm) -> None:
    """The runner ends quietly at its cap and hands back a turn whose tools never ran; an agent
    that stopped mid-task must not look like one that finished."""
    llm, fake = make_llm(turns(tool_turn(("t1", "current_time", {}))))

    with pytest.raises(TurnsExhausted, match="after 2 turns"):
        llm.run_tools(AGENT, execute_tool, max_turns=2)
    assert len(fake.sent) == 2


LONG = replace(AGENT, max_tokens=4096, cache_control={"type": "ephemeral"})
FINISH = "answer now from what you have"


def _context_costing(usd: float) -> int:
    """The cached context whose write alone costs `usd`: so a row reads relative to the ceiling."""
    return int(usd / (PRICES[MODEL].cache_write * 1e-6))


# Past the per-turn ceiling with the output cap on top, with $0.05 left for an answer.
GROWN_PAST = _context_costing(MAX_USD_PER_TURN - 0.05)


def _growing(fake: FakeMessages, tokens: int, ran: list[str] | None = None) -> Dispatch:
    """A tool whose result makes the next turn `tokens` long, by the free count."""

    def dispatch(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        fake.input_tokens = tokens
        if ran is not None:
            ran.append(name)
        return ("found", False)

    return dispatch


def _until_told(sent: Sent) -> Message | str:
    told = FINISH in str(sent.messages[-1]["content"])
    return "all I found" if told else tool_turn(("t1", "current_time", {}))


def test_a_turn_is_held_to_the_per_turn_ceiling_not_the_per_request_one(make_llm: MakeLlm) -> None:
    """A turn resends the whole conversation, so it has a ceiling of its own: $0.35 at worst is
    refused as a single call and admitted as a turn."""
    llm, fake = make_llm(lambda sent: "done")
    fake.input_tokens = 40_000

    with pytest.raises(BudgetExceeded, match="per-request ceiling"):
        llm.text(LONG)
    llm.run_tools(LONG, execute_tool)

    assert [sent.kind for sent in fake.sent] == ["turn"]


@pytest.mark.parametrize(
    ("grown_to", "max_turns", "says", "max_tokens"),
    [
        pytest.param(100, 2, "last of 2 turns", 4096, id="the last turn under the cap"),
        pytest.param(
            GROWN_PAST,
            8,
            "per-turn ceiling",
            affordable_output_tokens(MODEL, GROWN_PAST, MAX_USD_PER_TURN, cached=True),
            id="a search that grows the next turn past the per-turn ceiling",
        ),
    ],
)
def test_a_conversation_out_of_room_is_told_to_answer_from_what_it_has(
    make_llm: MakeLlm, grown_to: int, max_turns: int, says: str, max_tokens: int
) -> None:
    """Instead of stopping empty-handed, one last turn: `finish` after the tool results, and as
    much output as still fits — sent as the turns before it were, so the cache still holds."""
    llm, fake = make_llm(_until_told)
    tools_run: list[str] = []

    ran = llm.run_tools(
        LONG, _growing(fake, grown_to, tools_run), max_turns=max_turns, finish=FINISH
    )

    assert says in str(ran.cut_short), ran.cut_short
    assert len(tools_run) == len(fake.sent) - 1, "each tool turn's tools ran once"
    assert len(ran.turns) == len(fake.sent), "the last answer is one of the turns"
    before, told = fake.sent[-2:]
    same = ("model", "system", "thinking", "tools", "cache_control")
    assert [getattr(told, f) for f in same] == [getattr(before, f) for f in same]
    assert cast(list[object], told.messages[-1]["content"])[-1] == {"type": "text", "text": FINISH}
    assert told.max_tokens == max_tokens
    assert [block.text for block in ran.message.content if block.type == "text"] == ["all I found"]
    assert [call.step for call in llm.trace] == ["agent:1", "agent:2 answer"]


@pytest.mark.parametrize(
    "grown_to",
    [
        pytest.param(_context_costing(MAX_USD_PER_TURN) + 1_000, id="no room at all"),
        pytest.param(
            _context_costing(MAX_USD_PER_TURN - 0.01),
            id="room for less than MIN_TOKENS_FINAL_ANSWER",
        ),
    ],
)
def test_without_room_for_an_answer_the_refusal_stands(make_llm: MakeLlm, grown_to: int) -> None:
    llm, fake = make_llm(_until_told)

    with pytest.raises(BudgetExceeded, match="per-turn ceiling"):
        llm.run_tools(LONG, _growing(fake, grown_to), finish=FINISH)
    assert len(fake.sent) == 1, "no answer turn is sent"


def test_an_answer_turn_that_asks_for_tools_again_ends_the_run(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(lambda sent: tool_turn(("t1", "current_time", {})))

    with pytest.raises(TurnsExhausted, match="last of 2 turns"):
        llm.run_tools(LONG, execute_tool, max_turns=2, finish=FINISH)
    assert len(fake.sent) == len(llm.trace) == 2, "the answer turn was sent, and billed"


def test_server_tools_reach_the_wire_as_written_after_ours(make_llm: MakeLlm) -> None:
    """The runner wraps only the tools we run; one the API runs goes through untouched, last — on
    the last answer too, which must send the tools exactly as the runner did or miss the cache."""
    llm, fake = make_llm(_until_told)
    asked = replace(LONG, tools=[WEB_SEARCH_TOOL, *ALL_TOOLS])

    llm.run_tools(asked, execute_tool, max_turns=2, finish=FINISH)

    first, answer = (cast(list[dict[str, object]], sent.tools) for sent in fake.sent)
    assert [tool["name"] for tool in first] == [*(tool["name"] for tool in ALL_TOOLS), "web_search"]
    assert first[-1] == WEB_SEARCH_TOOL
    assert answer == first


def test_run_tools_needs_a_tool(make_llm: MakeLlm) -> None:
    llm, fake = make_llm(echo)

    with pytest.raises(ValueError, match="at least one tool"):
        llm.run_tools(REQUEST, execute_tool)
    assert fake.sent == [] and fake.counted == []
