"""The OpenAI wire: what it puts on the wire, what it reads back, and what it refuses.

Nothing here reaches a server, and nothing loads the served model's tokenizer: the scripted client
stands in for one and a stub for the other, as the Anthropic tests' fake does, so the conversion
either side is what is under test. A test that fetched a tokenizer would need the network to pass.
"""

from __future__ import annotations

from typing import cast

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import PromptTokensDetails
from pydantic import BaseModel

from entropic.adapters.cfg_local import CLIENT as LOCAL
from entropic.adapters.openai import OpenAI, record_in, usage_of
from entropic.errors import Unsupported
from entropic.llm import Llm, Request
from entropic.messages import Msg


class Verdict(BaseModel):
    passed: bool
    reason: str


def _completion(text: str, *, prompt: int = 100, completion: int = 20) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl_fake",
        object="chat.completion",
        created=0,
        model=LOCAL.model,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
    )


class ScriptedCompletions:
    def __init__(self, text: str) -> None:
        self._text = text
        self.sent: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> ChatCompletion:
        self.sent.append(kwargs)
        return _completion(self._text)


class ScriptedClient:
    """Stands in for `openai.OpenAI`, as far as this wire reaches into one."""

    def __init__(self, text: str) -> None:
        self.completions = ScriptedCompletions(text)

    @property
    def chat(self) -> ScriptedClient:
        return self


class StubTokenizer:
    """Stands in for the served model's tokenizer: a fixed count, and no download."""

    def apply_chat_template(self, conversation: object, **kwargs: object) -> list[int]:
        return list(range(42))


@pytest.fixture
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenAI, "tokenizer", property(lambda _: StubTokenizer()))


def _wire(text: str = "hello") -> tuple[OpenAI, ScriptedCompletions]:
    scripted = ScriptedClient(text)
    return OpenAI(LOCAL, scripted), scripted.completions


REQUEST = Request("step", [{"role": "user", "content": "the question"}], 64, system="be brief")


def test_the_system_prompt_becomes_the_first_message_and_blocks_are_flattened() -> None:
    blocks: list[Msg] = [{"role": "user", "content": [{"type": "text", "text": "a passage"}]}]
    wire, _ = _wire()
    sent = wire.messages(Request("step", blocks, 64, system=[{"type": "text", "text": "rules"}]))

    assert sent == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "a passage"},
    ]


def test_a_request_with_no_system_prompt_sends_no_system_message() -> None:
    wire, _ = _wire()
    assert wire.messages(Request("step", [{"role": "user", "content": "hi"}], 64)) == [
        {"role": "user", "content": "hi"}
    ]


def test_asking_for_a_record_appends_the_schema_to_the_system_prompt() -> None:
    """This wire has no constrained decoding, so the schema travels as an instruction."""
    wire, _ = _wire()
    system = str(cast(dict[str, object], wire.messages(REQUEST, Verdict)[0])["content"])

    assert system.startswith("be brief")
    assert "passed" in system and "reason" in system, "the schema's fields reach the model"
    assert "nothing else" in system


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param('{"passed": true, "reason": "it matches"}', True, id="bare JSON"),
        pytest.param(
            'Sure!\n```json\n{"passed": true, "reason": "it matches"}\n```', True, id="fenced"
        ),
        pytest.param("I could not say.", None, id="no JSON at all"),
        pytest.param('{"passed": "maybe"}', None, id="JSON the schema refuses"),
    ],
)
def test_a_record_is_read_out_of_whatever_the_model_wrapped_it_in(
    text: str, expected: bool | None
) -> None:
    record = record_in(text, Verdict)
    assert (record.passed if record else None) is expected, text


def test_the_count_is_the_templated_prompt_rather_than_an_estimate(offline_tokenizer: None) -> None:
    """Counting is what the per-request guard reads, so it counts the ids the server will see."""
    wire, _ = _wire()
    assert wire.count(REQUEST) == 42


def test_a_parsed_call_bills_its_usage_whether_or_not_a_record_came_back(
    offline_tokenizer: None,
) -> None:
    wire, _ = _wire("no JSON here")
    llm = Llm("local", sdk=cast(object, ScriptedClient("no JSON here")))
    parsed = llm.parse(REQUEST, Verdict)

    assert parsed.parsed is None, "prompting is not constraining; sometimes nothing parses"
    assert parsed.usage.input_tokens == 100 and parsed.usage.output_tokens == 20
    assert llm.trace[-1].usd > 0, "a local call costs machine time, which is not nothing"
    assert wire.name == "openai"


def test_cached_prompt_tokens_are_counted_as_read_not_as_fresh_input() -> None:
    usage = usage_of(
        CompletionUsage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=40),
        )
    )
    assert (usage.input_tokens, usage.cache_read_tokens) == (60, 40)


def test_a_server_that_reports_no_usage_is_billed_as_nothing_rather_than_guessed() -> None:
    assert usage_of(None) == usage_of(
        CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )


@pytest.mark.parametrize(
    ("request_", "refused"),
    [
        pytest.param(
            Request(
                "step", [{"role": "user", "content": "x"}], 64, cache_control={"type": "ephemeral"}
            ),
            "cache",
            id="a cache marker this wire cannot honour",
        ),
        pytest.param(
            Request(
                "step",
                [{"role": "user", "content": "x"}],
                64,
                thinking={"type": "enabled", "budget_tokens": 1024},
            ),
            "thinking",
            id="a thinking budget it has no field for",
        ),
        pytest.param(
            Request(
                "step", [{"role": "user", "content": "x"}], 64, output_config={"effort": "high"}
            ),
            "effort",
            id="an effort level it does not take",
        ),
        pytest.param(
            Request(
                "step",
                [{"role": "user", "content": "x"}],
                64,
                tools=[{"name": "search", "input_schema": {"type": "object"}}],
            ),
            "tools",
            id="tools, which no runner here drives yet",
        ),
    ],
)
def test_what_this_wire_cannot_do_is_refused_before_anything_is_sent(
    request_: Request, refused: str
) -> None:
    """The one failure a budget guard cannot catch: a wire that silently drops a field answers
    anyway, differently, and for a different price."""
    scripted = ScriptedClient("hi")
    llm = Llm("local", sdk=cast(object, scripted))

    with pytest.raises(Unsupported, match=refused):
        llm.text(request_)

    assert scripted.completions.sent == [], "nothing reached the server"
    assert llm.trace == [] and llm.spent_usd == 0.0
