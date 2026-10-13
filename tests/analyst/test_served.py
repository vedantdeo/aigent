"""The two specialists on scripted models: the prompt each adapter was trained on, the generalist as
the fallback, a dropped request sent once more, and the warm-up."""

from __future__ import annotations

from typing import cast

import httpx2 as httpx
import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall

from aigent.analyst.router import Router
from aigent.analyst.served import headline_reader, reaching, report_analyst, warm_up
from aigent.analyst.specialists import FigureJob
from aigent.config import SPECIALIST_TOOL_TURNS, SYNTHETIC_CLIENT, max_tokens
from aigent.errors import SpecialistUnreachable
from aigent.extraction.figures import TOOL as TRAINED_TOOL
from aigent.extraction.figures import figure_messages
from aigent.extraction.headlines import VARIANTS, Extraction, headline_request
from aigent.extraction.synthetic import wire_for
from aigent.llm.adapters import spec
from aigent.llm.core import Llm
from aigent.llm.pricing import cost_usd

from ..conftest import (
    FakeOpenAI,
    MakeLlm,
    Sent,
    calling,
    completion,
    tool_results,
    tool_turn,
    turns,
)

HEADLINE = "Infy Q2 profit rises 11% on deal wins"
RECORD = Extraction(company="Infy", metric="profit", quarter="Q2", direction="up", change_pct=11.0)
GARBLED = RECORD.model_copy(update={"metric": "growth"})
JOB = FigureJob(
    passage="Revenue from operations stood at ₹ 2,345 crore; the EBITDA margin was 18%.",
    source="ITC-FY25",
    wanted="revenue and margin",
)
FIGURES = (
    '{"figures": ['
    '{"metric": "revenue", "value": 23450000000.0, "unit": "INR", "year": 2025, '
    '"description": null}, '
    '{"metric": "margin", "value": 0.18, "unit": null, "year": 2025, "description": null}]}'
)
REQUEST = httpx.Request("POST", "https://hostllm.example/v1/chat/completions")
DROPPED = openai.APIConnectionError(request=REQUEST)


def small(client: str, *replies: object) -> tuple[Llm, FakeOpenAI]:
    fake = FakeOpenAI(*cast("tuple[ChatCompletion | Exception, ...]", replies))
    return Llm(client, sdk=cast(object, fake)), fake


def big(make_llm: MakeLlm, reply: object) -> Llm:
    return make_llm(reply)[0]


def router() -> Router:
    return Router(price=lambda i, o: cost_usd("claude-sonnet-5", i, o))


def test_a_headline_reaches_the_adapter_as_a_training_row_did(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, fake = small("hostllm-headline", completion(RECORD.model_dump_json()))

    told = headline_reader(llm, big(make_llm, "unused")).ask(HEADLINE)

    [sent] = fake.sent
    trained = headline_request("row", HEADLINE, VARIANTS["zero_shot"], 0)
    assert sent["messages"] == wire_for(SYNTHETIC_CLIENT).messages(trained, Extraction)
    assert sent["model"] == "headline-flavour"
    assert sent["max_tokens"] == max_tokens("HEADLINE", spec("hostllm-headline"))
    assert (told.input_tokens, told.output_tokens) == (100, 20)
    assert told.usd == pytest.approx(cost_usd("headline-flavour", 100, 20)) and told.usd > 0


@pytest.mark.parametrize(
    ("served", "path"),
    [
        pytest.param(RECORD, "specialist", id="a record that passes is the adapter's"),
        pytest.param(GARBLED, "generalist", id="one off the vocabulary goes to the generalist"),
    ],
)
def test_a_routed_headline_comes_back_with_its_ticker(
    make_llm: MakeLlm, offline_tokenizer: None, served: Extraction, path: str
) -> None:
    llm, _ = small("hostllm-headline", completion(served.model_dump_json()))
    route = router()

    record = route(headline_reader(llm, big(make_llm, lambda sent: RECORD)), HEADLINE)

    assert (record["ticker"], record["metric"]) == ("INFY", "profit")
    [logged] = route.routed
    assert logged.path == path and (logged.generalist_usd > 0) is (path == "generalist")


def test_figures_reach_the_adapter_on_its_trained_prompt_and_its_calls_are_run(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, fake = small(
        "hostllm-analyst", calling("2345 * 10**7", "18 / 100"), completion(FIGURES, prompt=140)
    )
    route = router()

    figures = route(report_analyst(llm, big(make_llm, "unused")), JOB)

    system, user = figure_messages(JOB.passage, JOB.source, JOB.wanted)
    first, second = fake.sent
    assert first["messages"] == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    assert first["tools"] == [TRAINED_TOOL] and first["model"] == "report-analyst"
    results = [m for m in cast(list[dict[str, object]], second["messages"]) if m["role"] == "tool"]
    assert [m["content"] for m in results] == ["23450000000.0", "0.18"]
    assert [f.value for f in figures.figures] == [23450000000.0, 0.18]
    [logged] = route.routed
    assert logged.path == "specialist"
    assert logged.specialist_usd == pytest.approx(cost_usd("report-analyst", 240, 40))


def test_figures_the_passage_does_not_back_go_to_the_generalist(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    wrong = FIGURES.replace("0.18", "0.19")
    llm, _ = small("hostllm-analyst", calling("2345 * 10**7", "18 / 100"), completion(wrong))
    asked = tool_turn(
        ("c1", "calculate", {"expression": "2345 * 10**7"}),
        ("c2", "calculate", {"expression": "18 / 100"}),
    )
    generalist, sdk = make_llm(turns(asked, FIGURES))
    route = router()

    figures = route(report_analyst(llm, generalist), JOB)

    assert [f.value for f in figures.figures] == [23450000000.0, 0.18]
    assert route.routed[0].path == "generalist" and "0.19" in str(route.routed[0].failure)
    sent: Sent = sdk.sent[1]
    assert [r["content"] for r in tool_results(sent)] == ["23450000000.0", "0.18"]


def test_a_call_to_any_tool_but_calculate_is_refused_and_runs_nothing(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    reading = calling("../.env")
    calls = reading.choices[0].message.tool_calls
    assert calls is not None
    function = cast(ChatCompletionMessageFunctionToolCall, calls[0]).function
    function.name, function.arguments = "read_file", '{"file_path": "notes.txt"}'
    llm, fake = small("hostllm-analyst", reading, completion('{"figures": []}'))

    told = report_analyst(llm, big(make_llm, "unused")).ask(JOB)

    [result] = [
        m for m in cast(list[dict[str, object]], fake.sent[1]["messages"]) if m["role"] == "tool"
    ]
    assert result["content"] == "Error: read_file is not available here" and told.calls == ()


def test_a_model_still_calculating_after_its_turns_has_no_answer_to_check(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, fake = small("hostllm-analyst", calling("18 / 100"))

    told = report_analyst(llm, big(make_llm, "unused")).ask(JOB)

    assert told.text == "" and len(fake.sent) == SPECIALIST_TOOL_TURNS
    assert told.calls == ("18 / 100",) * SPECIALIST_TOOL_TURNS


@pytest.mark.parametrize(
    "dropped",
    [
        pytest.param(DROPPED, id="a dropped connection"),
        pytest.param(openai.APITimeoutError(request=REQUEST), id="a timeout"),
        pytest.param(
            openai.InternalServerError(
                "bad gateway", response=httpx.Response(502, request=REQUEST), body=None
            ),
            id="a 502 from in front of it",
        ),
    ],
)
def test_a_request_the_endpoint_drops_is_sent_once_more(
    make_llm: MakeLlm, offline_tokenizer: None, dropped: Exception
) -> None:
    llm, fake = small("hostllm-headline", dropped, completion(RECORD.model_dump_json()))

    told = headline_reader(llm, big(make_llm, "unused")).ask(HEADLINE)

    assert len(fake.sent) == 2 and '"Infy"' in told.text


def test_figures_whose_request_is_dropped_are_asked_for_once_more(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, fake = small("hostllm-analyst", DROPPED, completion('{"figures": []}'))

    told = report_analyst(llm, big(make_llm, "unused")).ask(JOB)

    assert len(fake.sent) == 2 and told.text == '{"figures": []}'


def test_an_endpoint_that_stays_down_is_unreachable_and_the_generalist_is_not_asked(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, fake = small("hostllm-headline", DROPPED)
    generalist, sdk = make_llm(lambda sent: RECORD)
    route = router()

    with pytest.raises(SpecialistUnreachable, match="hostllm-headline did not answer in 2 tries"):
        route(headline_reader(llm, generalist), HEADLINE)
    assert len(fake.sent) == 2 and sdk.sent == []
    assert route.routed[0].path is None and "unreachable" in str(route.routed[0].failure)


def test_a_refusal_that_is_not_the_endpoint_being_down_is_not_retried(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    denied = openai.AuthenticationError(
        "bad proxy token", response=httpx.Response(401, request=REQUEST), body=None
    )
    llm, fake = small("hostllm-headline", denied)

    with pytest.raises(openai.AuthenticationError):
        headline_reader(llm, big(make_llm, "unused")).ask(HEADLINE)
    assert len(fake.sent) == 1


def test_the_warm_up_asks_the_endpoint_for_one_token(offline_tokenizer: None) -> None:
    llm, fake = small("hostllm-headline", DROPPED, completion("ok", completion=1))

    warm_up(llm)

    assert [sent["max_tokens"] for sent in fake.sent] == [1, 1], "sent again after a drop"
    assert fake.sent[0]["model"] == "headline-flavour" and llm.spent_usd > 0


def test_reaching_gives_up_after_its_tries(offline_tokenizer: None) -> None:
    llm, _ = small("hostllm-analyst", DROPPED)
    tried: list[int] = []

    def call() -> None:
        tried.append(1)
        raise DROPPED

    with pytest.raises(SpecialistUnreachable, match="hostllm-analyst"):
        reaching(llm, call)
    assert len(tried) == 2
