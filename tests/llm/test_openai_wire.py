"""The OpenAI wire: what it puts on the wire, what it reads back, and what it refuses.

Nothing here reaches a server, and nothing loads the served model's tokenizer: the scripted client
stands in for one and a stub for the other, as the Anthropic tests' fake does, so the conversion
either side is what is under test. A test that fetched a tokenizer would need the network to pass.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from openai import omit
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.completion_usage import PromptTokensDetails
from pydantic import BaseModel

from aigent.errors import StepFailed, Unsupported
from aigent.extraction.figures import TOOL as TRAINED_TOOL
from aigent.llm.adapters.cfg_hostllm import ANALYST, COLD_START_TIMEOUT, HEADLINE
from aigent.llm.adapters.cfg_local import CLIENT as LOCAL
from aigent.llm.adapters.cfg_sarvam import CLIENT as SARVAM
from aigent.llm.adapters.client import Client
from aigent.llm.adapters.openai import LOCAL_API_KEY, OpenAI, record_in, reply_of, usage_of
from aigent.llm.core import Dispatch, Llm, Request
from aigent.llm.messages import Msg
from aigent.tools import CALCULATOR_TOOL

from ..conftest import FakeOpenAI, Finish, StubTokenizer
from ..conftest import calling as _calling
from ..conftest import completion as _completion


class Verdict(BaseModel):
    passed: bool
    reason: str


class ScriptedCompletions:
    def __init__(self, text: str) -> None:
        self._text = text
        self.finish: Finish = "stop"
        self.sent: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> ChatCompletion:
        self.sent.append(kwargs)
        return _completion(self._text, finish=self.finish)


class ScriptedClient:
    """Stands in for `openai.OpenAI`, as far as this wire reaches into one."""

    def __init__(self, text: str) -> None:
        self.completions = ScriptedCompletions(text)

    @property
    def chat(self) -> ScriptedClient:
        return self


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


def test_a_hosted_clients_own_fields_ride_beside_the_servers() -> None:
    scripted = ScriptedClient("hi")
    OpenAI(replace(SARVAM, template_kwargs={"enable_thinking": False}), scripted).send(REQUEST)

    assert scripted.completions.sent[0]["extra_body"] == {
        "reasoning_effort": None,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_a_client_whose_server_enforces_schemas_is_sent_the_records() -> None:
    scripted = ScriptedClient('{"passed": true, "reason": "fine"}')
    OpenAI(SARVAM, scripted).parse(REQUEST, Verdict)

    assert scripted.completions.sent[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Verdict", "schema": Verdict.model_json_schema()},
    }


def test_a_server_that_only_follows_the_prompt_is_sent_no_schema_to_enforce() -> None:
    """mlx_lm.server takes no `response_format`; a local record stays prompted."""
    scripted = ScriptedClient('{"passed": true, "reason": "fine"}')
    OpenAI(LOCAL, scripted).parse(REQUEST, Verdict)

    assert scripted.completions.sent[0]["response_format"] is omit


def test_the_output_cap_is_sent_under_the_name_every_server_here_reads() -> None:
    """Sarvam documents `max_tokens` only; mlx_lm.server reads either name."""
    scripted = ScriptedClient("hi")
    OpenAI(LOCAL, scripted).send(REQUEST)

    assert scripted.completions.sent[0]["max_tokens"] == REQUEST.max_tokens
    assert "max_completion_tokens" not in scripted.completions.sent[0]


@pytest.mark.parametrize(
    ("client", "environment", "key"),
    [
        pytest.param(LOCAL, {}, LOCAL_API_KEY, id="a local server takes a placeholder"),
        pytest.param(SARVAM, {"SARVAM_API_KEY": "sk_real"}, "sk_real", id="a hosted key is read"),
    ],
)
def test_the_key_comes_from_the_variable_the_client_names(
    monkeypatch: pytest.MonkeyPatch, client: Client, environment: dict[str, str], key: str
) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert OpenAI(client).client.api_key == key


def test_a_hosted_client_without_its_key_is_refused_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):
        _ = OpenAI(SARVAM).client


HOSTLLM_ENV = {
    "HOSTLLM_API_KEY": "sk_host",
    "HOSTLLM_MODAL_KEY": "wk-id",
    "HOSTLLM_MODAL_SECRET": "ws-secret",
}


def test_a_gateway_s_headers_come_from_the_variables_the_client_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in HOSTLLM_ENV.items():
        monkeypatch.setenv(name, value)

    client = OpenAI(HEADLINE).client

    assert client.api_key == "sk_host" and str(client.base_url).endswith("modal.run/v1/")
    assert client.default_headers["Modal-Key"] == "wk-id"
    assert client.default_headers["Modal-Secret"] == "ws-secret"
    assert "Modal-Key" not in OpenAI(LOCAL).client.default_headers


@pytest.mark.parametrize("unset", ["HOSTLLM_MODAL_KEY", "HOSTLLM_MODAL_SECRET"])
def test_a_client_without_a_header_s_variable_is_refused_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch, unset: str
) -> None:
    for name, value in HOSTLLM_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(unset)

    with pytest.raises(RuntimeError, match=unset):
        _ = OpenAI(HEADLINE).client


def test_the_two_specialists_share_an_endpoint_and_differ_in_the_adapter_they_name() -> None:
    assert (HEADLINE.model, ANALYST.model) == ("headline-flavour", "report-analyst")
    assert replace(ANALYST, name="", model="", small_model="") == replace(
        HEADLINE, name="", model="", small_model=""
    )
    assert HEADLINE.timeout == COLD_START_TIMEOUT >= 180, "a request outlasts a cold start"


@pytest.mark.parametrize(
    ("client", "repo"),
    [
        pytest.param(LOCAL, LOCAL.model, id="a local model is its own tokenizer's repo"),
        pytest.param(HEADLINE, "Qwen/Qwen3-4B", id="an adapter counts on its base's tokenizer"),
        pytest.param(SARVAM, "sarvamai/sarvam-105b", id="a hosted id names the repo apart"),
    ],
)
def test_counting_loads_the_tokenizer_the_client_names_without_remote_code(
    monkeypatch: pytest.MonkeyPatch, client: Client, repo: str
) -> None:
    import transformers

    loaded: list[tuple[str, dict[str, object]]] = []

    def load(name: str, **kwargs: object) -> StubTokenizer:
        loaded.append((name, kwargs))
        return StubTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    OpenAI(client).count(REQUEST)

    assert loaded == [(repo, {"trust_remote_code": False})]


@pytest.mark.parametrize(
    ("finish", "stop"),
    [
        pytest.param("length", "max_tokens", id="a cap that cut the reply off"),
        pytest.param("content_filter", "refusal", id="a filtered reply"),
        pytest.param("tool_calls", "tool_use", id="a turn that asked for tools"),
        pytest.param("stop", "stop", id="a finished reply passes through"),
    ],
)
def test_a_finish_reason_arrives_under_the_name_llm_checks(finish: Finish, stop: str) -> None:
    assert reply_of(_completion("", finish=finish)).stop_reason == stop


def test_a_reply_cut_off_by_its_cap_fails_rather_than_reading_as_an_empty_answer() -> None:
    """Sarvam's first call reasoned through its whole cap and wrote nothing; it printed as ''."""
    scripted = ScriptedClient("")
    scripted.completions.finish = "length"
    llm = Llm("local", sdk=cast(object, scripted))

    with pytest.raises(StepFailed, match="max_tokens"):
        llm.text(REQUEST)

    assert llm.spent_usd > 0, "the truncated reply is still billed"


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
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
            ),
            "web_search",
            id="a tool only the provider can run",
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


ASKING = Request("step", [{"role": "user", "content": "18% of what?"}], 64, tools=[CALCULATOR_TOOL])


def test_a_tool_is_sent_as_the_function_the_fine_tune_was_trained_on() -> None:
    wire, completions = _wire()

    wire.send(ASKING)

    assert completions.sent[0]["tools"] == [TRAINED_TOOL]


def test_a_request_with_no_tools_sends_none() -> None:
    wire, completions = _wire()

    wire.send(REQUEST)

    assert completions.sent[0]["tools"] is omit


def test_the_calls_a_reply_asks_for_come_back_as_tool_use_blocks() -> None:
    reply = reply_of(_calling("18 / 100", "2345 * 10**7"))

    assert reply.stop_reason == "tool_use" and reply.text == ""
    assert list(reply.blocks) == [
        {
            "type": "tool_use",
            "id": "call_0",
            "name": "calculate",
            "input": {"expression": "18 / 100"},
        },
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "calculate",
            "input": {"expression": "2345 * 10**7"},
        },
    ]


def test_text_beside_a_call_is_kept_ahead_of_it() -> None:
    reply = reply_of(_calling("18 / 100", text="Converting."))

    assert [block["type"] for block in reply.blocks] == ["text", "tool_use"]
    assert reply.text == "Converting."


def test_a_call_whose_arguments_are_not_json_arrives_with_none() -> None:
    broken = _calling("18 / 100")
    calls = broken.choices[0].message.tool_calls
    assert calls is not None
    cast(ChatCompletionMessageFunctionToolCall, calls[0]).function.arguments = "{not json"

    assert reply_of(broken).blocks[0]["input"] == {}


def test_a_turn_s_calls_and_their_results_go_back_as_the_wire_carries_them() -> None:
    wire, completions = _wire()
    asked = reply_of(_calling("18 / 100", text="Converting."))
    results: Msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_0", "content": "0.18", "is_error": False}
        ],
    }
    messages: list[Msg] = [
        *ASKING.messages,
        {"role": "assistant", "content": asked.blocks},
        results,
    ]

    wire.send(replace(ASKING, messages=messages))

    assert completions.sent[0]["messages"] == [
        {"role": "user", "content": "18% of what?"},
        {
            "role": "assistant",
            "content": "Converting.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": '{"expression": "18 / 100"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "0.18"},
    ]


def test_the_count_templates_the_tools_too(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    class Recording(StubTokenizer):
        def apply_chat_template(
            self, conversation: object, **kwargs: object
        ) -> list[int] | dict[str, list[int]]:
            seen.append(kwargs.get("tools"))
            return super().apply_chat_template(conversation, **kwargs)

    monkeypatch.setattr(OpenAI, "tokenizer", property(lambda _: Recording()))
    wire, _ = _wire()

    wire.count(ASKING)
    wire.count(REQUEST)

    assert seen == [[TRAINED_TOOL], None]


def test_the_count_hands_a_template_each_call_s_arguments_as_a_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    class Recording(StubTokenizer):
        def apply_chat_template(
            self, conversation: object, **kwargs: object
        ) -> list[int] | dict[str, list[int]]:
            seen.extend(cast(list[object], conversation))
            return super().apply_chat_template(conversation, **kwargs)

    monkeypatch.setattr(OpenAI, "tokenizer", property(lambda _: Recording()))
    wire, completions = _wire()
    asked: Msg = {"role": "assistant", "content": reply_of(_calling("18 / 100")).blocks}
    request = replace(ASKING, messages=[*ASKING.messages, asked])

    wire.count(request)
    wire.send(request)

    [_, counted] = cast(list[dict[str, list[dict[str, dict[str, object]]]]], seen)
    assert counted["tool_calls"][0]["function"]["arguments"] == {"expression": "18 / 100"}
    [_, sent] = cast(
        list[dict[str, list[dict[str, dict[str, object]]]]], completions.sent[0]["messages"]
    )
    assert sent["tool_calls"][0]["function"]["arguments"] == '{"expression": "18 / 100"}'


def test_llm_sends_one_tool_turn_on_this_wire_and_bills_it(offline_tokenizer: None) -> None:
    scripted = ScriptedClient("hi")
    scripted.completions.create = lambda **kwargs: _calling("18 / 100")  # type: ignore[method-assign]
    llm = Llm("local", sdk=cast(object, scripted))

    reply = llm.turn(ASKING)

    assert reply.stop_reason == "tool_use" and reply.blocks[0]["name"] == "calculate"
    assert len(llm.trace) == 1


def _dispatch(ran: list[str]) -> Dispatch:
    """A calculate that logs each expression and answers with its length; `1 / 0` is an error."""

    def dispatch(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        expression = str(arguments["expression"])
        ran.append(expression)
        return str(len(expression)), expression == "1 / 0"

    return dispatch


def test_llm_loops_tools_on_this_wire_running_each_call_once(offline_tokenizer: None) -> None:
    fake = FakeOpenAI(_calling("18 / 100", "1 / 0"), _completion("It is 0.18."))
    llm = Llm("local", sdk=cast(object, fake))
    ran: list[str] = []

    done = llm.run_tools(ASKING, _dispatch(ran))

    assert done.message.text == "It is 0.18." and done.cut_short is None and len(done.turns) == 2
    assert ran == ["18 / 100", "1 / 0"], "each call run once, not again when the loop resumes"
    assert len(llm.trace) == 2 and len(fake.sent) == 2, "every turn sent once and billed"
    first, second = (cast(list[dict[str, object]], sent["messages"]) for sent in fake.sent)
    assert [m["role"] for m in second] == ["user", "assistant", "tool", "tool"]
    assert second[0] == first[0] and second[2:] == [
        {"role": "tool", "tool_call_id": "call_0", "content": "8"},
        {"role": "tool", "tool_call_id": "call_1", "content": "5"},
    ]
    assert all(sent["tools"] == [TRAINED_TOOL] for sent in fake.sent)


def test_a_loop_out_of_turns_is_told_to_answer_after_its_tool_results(
    offline_tokenizer: None,
) -> None:
    fake = FakeOpenAI(_calling("18 / 100"), _completion("About 0.18."))
    llm = Llm("local", sdk=cast(object, fake))
    roomy = replace(ASKING, max_tokens=2048)

    done = llm.run_tools(roomy, _dispatch([]), max_turns=2, finish="Answer now.")

    assert done.message.text == "About 0.18." and done.cut_short is not None
    assert cast(list[dict[str, object]], fake.sent[1]["messages"])[-2:] == [
        {"role": "tool", "tool_call_id": "call_0", "content": "8"},
        {"role": "user", "content": "Answer now."},
    ]


def test_a_session_stops_asking_at_its_turn_cap() -> None:
    fake = FakeOpenAI(_calling("18 / 100"))
    ran: list[str] = []

    replies = list(OpenAI(LOCAL, fake).tools(ASKING, _dispatch(ran), 3))

    assert len(replies) == len(fake.sent) == len(ran) == 3
