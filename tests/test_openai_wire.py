"""The OpenAI wire: what it puts on the wire, what it reads back, and what it refuses.

Nothing here reaches a server, and nothing loads the served model's tokenizer: the scripted client
stands in for one and a stub for the other, as the Anthropic tests' fake does, so the conversion
either side is what is under test. A test that fetched a tokenizer would need the network to pass.
"""

from __future__ import annotations

from dataclasses import replace
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
    """Stands in for the served model's tokenizer, including the shape it answers in.

    transformers v5 hands back a `BatchEncoding` unless `return_dict=False`, so this returns a
    mapping too — a count taken from its length would be 2 for any prompt, which is the bug this
    shape exists to catch.
    """

    def apply_chat_template(
        self, conversation: object, **kwargs: object
    ) -> list[int] | dict[str, list[int]]:
        ids = list(range(42))
        if kwargs.get("return_dict") is False:
            return ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


@pytest.fixture
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenAI, "tokenizer", property(lambda _: StubTokenizer()))


def _wire(text: str = "hello") -> tuple[OpenAI, ScriptedCompletions]:
    scripted = ScriptedClient(text)
    return OpenAI(LOCAL, scripted), scripted.completions


REQUEST = Request("step", [{"role": "user", "content": "the question"}], 64, system="be brief")


@pytest.mark.parametrize(
    ("adapter", "sent"),
    [
        pytest.param(None, None, id="a client with no adapter sends none"),
        pytest.param("/adapters/toy", "/adapters/toy", id="a client's adapter rides with the call"),
    ],
)
def test_the_server_is_told_which_adapter_to_apply(adapter: str | None, sent: str | None) -> None:
    """mlx_lm.server reads `adapters` from each request, so one server answers as both models."""
    scripted = ScriptedClient('{"passed": true, "reason": "fine"}')
    wire = OpenAI(replace(LOCAL, adapter=adapter), scripted)

    wire.parse(REQUEST, Verdict)

    body = cast(dict[str, object], scripted.completions.sent[0]["extra_body"])
    assert body.get("adapters") == sent
    assert body["chat_template_kwargs"] == dict(LOCAL.template_kwargs), "templating still rides"


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


def test_asking_for_a_record_describes_the_fields_rather_than_sending_a_schema() -> None:
    """This wire has no constrained decoding, so the shape travels as an instruction — as a list
    of keys, never as a JSON Schema document. Handing Qwen3-8B a schema made it answer *with* the
    schema: 0 of 12 replies parsed, against 12 of 12 for this wording.
    """
    wire, _ = _wire()
    system = str(cast(dict[str, object], wire.messages(REQUEST, Verdict)[0])["content"])

    assert system.startswith("be brief")
    assert "passed (boolean)" in system and "reason (string)" in system, "the fields, in words"
    assert "nothing else" in system
    assert '"type": "object"' not in system and '"properties"' not in system, "no schema document"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param('{"passed": true, "reason": "it matches"}', True, id="bare JSON"),
        pytest.param(
            'Sure!\n```json\n{"passed": true, "reason": "it matches"}\n```', True, id="fenced"
        ),
        pytest.param(
            '{"title": "Verdict", "type": "object", "properties": '
            '{"passed": true, "reason": "it matches"}, "required": ["passed", "reason"]}',
            True,
            id="the schema echoed back with the values inside it",
        ),
        pytest.param(
            '{"title": "Verdict", "description": "a verdict", "properties": '
            '{"passed": true, "reason": "it matches"}, "required": ["passed", "reason"]}',
            True,
            id="the same echo with the type key dropped, which is what a model actually sends",
        ),
        pytest.param(
            '{"title": "Verdict", "type": "object", "properties": '
            '{"passed": {"type": "boolean"}, "reason": {"type": "string"}}}',
            None,
            id="the schema echoed back with no values in it at all",
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
    """Counting is what the per-request guard reads, so it counts the ids the server will see.

    A live call found this returning 2 for every prompt: the tokenizer answers with a mapping of
    two fields unless asked otherwise, and a token count that is always 2 waves everything past
    the ceiling it is supposed to hold.
    """
    wire, _ = _wire()
    assert wire.count(REQUEST) == 42


def test_a_request_is_sent_to_the_model_its_own_client_serves(offline_tokenizer: None) -> None:
    """A live call found `Llm("local")` sending the configured *other* client's model, because the
    default came from `config.MODEL` rather than from the client being talked to."""
    scripted = ScriptedClient("hi")
    Llm("local", sdk=cast(object, scripted)).text(REQUEST)

    assert scripted.completions.sent[0]["model"] == LOCAL.model


def test_a_request_that_names_a_model_keeps_it(offline_tokenizer: None) -> None:
    """Which is how the routing workflow reaches for a smaller model on the same client."""
    scripted = ScriptedClient("hi")
    asked = Request("step", [{"role": "user", "content": "x"}], 64, model=LOCAL.small_model)
    Llm("local", sdk=cast(object, scripted)).text(asked)

    assert scripted.completions.sent[0]["model"] == LOCAL.small_model


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


def test_a_clients_template_arguments_ride_along_with_every_request(
    offline_tokenizer: None,
) -> None:
    """A hybrid-thinking model spends its output cap on thought unless told not to, and then no
    record is reached. Qwen3's models are hybrid; Qwen2.5's are not, and the 16 rows a 7B lost to
    unparseable output were a different failure — it ignored the schema and wrote prose. Two
    causes, one symptom, and only this one is ours to prevent.
    """
    scripted = ScriptedClient("hi")
    Llm("local-8b-4bit", sdk=cast(object, scripted)).text(REQUEST)

    assert scripted.completions.sent[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_a_client_with_nothing_to_say_about_templating_sends_no_extra_body() -> None:
    """Every local client happens to carry the thinking flag, so this builds one that does not:
    an empty mapping must send nothing rather than an empty `chat_template_kwargs`."""
    scripted = ScriptedClient("hi")
    plain = replace(LOCAL, template_kwargs={})
    OpenAI(plain, scripted).send(REQUEST)

    assert scripted.completions.sent[0]["extra_body"] is None


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
