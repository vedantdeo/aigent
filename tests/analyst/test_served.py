"""The report analyst on scripted models: the prompt and tools its adapter is trained on, its
searches held to the one report asked about, the generalist as the fallback, a dropped request sent
once more, and the warm-up."""

from __future__ import annotations

from typing import cast

import httpx2 as httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from aigent.analyst.cfg_analyst import GENERALIST_MODEL, MAX_TOKENS, SPECIALIST_TOOL_TURNS
from aigent.analyst.router import Router
from aigent.analyst.served import reaching, report, report_analyst, research, warm_up
from aigent.analyst.specialists import FigureJob, canonical
from aigent.config import CACHE_PARAM
from aigent.errors import SpecialistUnreachable
from aigent.extraction.figures import TOOL as TRAINED_TOOL
from aigent.extraction.research import REPORT_SEARCH_TOOL, research_messages
from aigent.llm.adapters.cfg_hostllm import ANALYST, RESEARCH_TURN_HOSTLLM
from aigent.llm.adapters.openai import function_tool
from aigent.llm.core import Llm
from aigent.llm.messages import Cache, Usage
from aigent.llm.pricing import cost_usd
from aigent.retrieval.chunk import Chunk, context_block
from aigent.tools import fence

from ..conftest import (
    FakeOpenAI,
    MakeLlm,
    Sent,
    asking,
    calling,
    completion,
    tool_results,
    tool_turn,
    turns,
)

JOB = FigureJob("What revenue and EBITDA margin did ITC report for FY25?", "ITC", 2025)
PASSAGE = Chunk(
    id="ITC-FY25#0007",
    text="Revenue from operations stood at ₹ 2,345 crore; the EBITDA margin was 18%.",
    doc_id="ITC-FY25",
    ordinal=7,
    start=0,
    page=12,
)
FIGURES = (
    '{"figures": ['
    '{"metric": "revenue", "value": 23450000000.0, "unit": "INR", "year": 2025, '
    '"description": null, "chunk_id": "ITC-FY25#0007"}, '
    '{"metric": "margin", "value": 0.18, "unit": null, "year": 2025, "description": null, '
    '"chunk_id": "ITC-FY25#0007"}]}'
)
SEARCHING = asking(("search_report", {"query": "revenue from operations", "company": "INFY"}))
CALCULATING = calling("2345 * 10**7", "18 / 100")
REQUEST = httpx.Request("POST", "https://hostllm.example/v1/chat/completions")
DROPPED = openai.APIConnectionError(request=REQUEST)


class Served:
    """The served `search_filings`: the one passage, fenced as the server fences it; every call
    logged."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, dict[str, object]]] = []

    def __call__(self, name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        self.asked.append((name, arguments))
        return fence("search_filings", context_block([PASSAGE])), False


def small(client: str, *replies: object) -> tuple[Llm, FakeOpenAI]:
    fake = FakeOpenAI(*cast("tuple[ChatCompletion | Exception, ...]", replies))
    return Llm(client, sdk=cast(object, fake)), fake


def big(make_llm: MakeLlm, reply: object) -> Llm:
    return make_llm(reply)[0]


def router() -> Router:
    return Router(price=lambda i, o: cost_usd(GENERALIST_MODEL, i, o))


def test_a_question_reaches_the_adapter_on_its_trained_prompt_and_tools(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, fake = small("hostllm-analyst", SEARCHING, CALCULATING, completion(FIGURES, prompt=140))
    served, route = Served(), router()

    found = route(report_analyst(llm, big(make_llm, "unused"), served), JOB)

    system, user = research_messages(JOB.question, "ITC Limited", "Annual report 2024-25")
    first, _, third = fake.sent
    assert first["messages"] == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    assert first["tools"] == [function_tool(REPORT_SEARCH_TOOL), TRAINED_TOOL]
    assert first["model"] == "report-analyst"
    assert first["max_tokens"] == ANALYST.max_tokens["RESEARCH_TURN"] == RESEARCH_TURN_HOSTLLM
    results = [m for m in cast(list[dict[str, object]], third["messages"]) if m["role"] == "tool"]
    assert PASSAGE.text in str(results[0]["content"]), "the search's passages went back"
    assert [m["content"] for m in results[1:]] == ["23450000000.0", "0.18"]
    assert [f.value for f in found.figures] == [23450000000.0, 0.18]
    assert list(found.passages) == [PASSAGE.id] and found.passages[PASSAGE.id].text == PASSAGE.text
    [logged] = route.routed
    assert logged.path == "specialist"
    assert logged.specialist_usd == pytest.approx(cost_usd("report-analyst", 340, 60))


def test_a_search_reads_only_the_report_the_job_names_whatever_the_model_asks(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, _ = small("hostllm-analyst", SEARCHING, completion('{"figures": []}'))
    served = Served()

    told = report_analyst(llm, big(make_llm, "unused"), served).ask(JOB)

    assert served.asked == [
        ("search_filings", {"query": "revenue from operations", "company": "ITC", "year": "2025"})
    ]
    assert [p.id for p in told.read] == [PASSAGE.id]


@pytest.mark.parametrize(
    ("year", "named"),
    [
        (2025, "Annual report 2024-25"),
        (2024, "Annual report 2023-24"),
        (2010, "Annual report 2009-10"),
    ],
)
def test_a_report_is_named_by_the_years_its_financial_year_spans(year: int, named: str) -> None:
    assert report(year) == named


def test_figures_their_passage_does_not_back_go_to_the_generalist(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    wrong = FIGURES.replace("0.18", "0.19")
    llm, _ = small("hostllm-analyst", SEARCHING, CALCULATING, completion(wrong))
    searched = tool_turn(("s1", "search_report", {"query": "revenue"}))
    calculated = tool_turn(
        ("c1", "calculate", {"expression": "2345 * 10**7"}),
        ("c2", "calculate", {"expression": "18 / 100"}),
    )
    generalist, sdk = make_llm(turns(searched, calculated, FIGURES))
    served, route = Served(), router()

    found = route(report_analyst(llm, generalist, served), JOB)

    assert [f.value for f in found.figures] == [23450000000.0, 0.18]
    assert route.routed[0].path == "generalist" and "0.19" in str(route.routed[0].failure)
    sent: Sent = sdk.sent[2]
    assert sent.model == GENERALIST_MODEL
    assert [r["content"] for r in tool_results(sent)] == ["23450000000.0", "0.18"]
    assert [tool["name"] for tool in cast(list[dict[str, object]], sent.tools)] == [
        "search_report",
        "calculate",
    ]
    assert len(served.asked) == 2, "each model searched for itself"
    assert sent.max_tokens == MAX_TOKENS["RESEARCH_TURN"], "the analyst's cap, not the default"


@pytest.mark.parametrize("cache", [CACHE_PARAM, None], ids=["cached", "not cached"])
def test_every_turn_is_kept_in_the_conversation_and_marked_for_a_cache_if_asked(
    make_llm: MakeLlm, cache: Cache | None
) -> None:
    searched = tool_turn(("s1", "search_report", {"query": "revenue"}))
    llm, sdk = make_llm(turns(searched, '{"figures": []}'))

    told = research(llm, Served(), GENERALIST_MODEL, cache=cache)(JOB)

    roles = [m["role"] for m in told.conversation]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert (
        told.conversation[0]["content"]
        == research_messages(JOB.question, "ITC Limited", "Annual report 2024-25")[1]
    )
    assert [sent.cache_control for sent in sdk.sent] == [cache, cache]


def test_a_job_records_its_own_turns_cost_not_what_others_billed_the_same_llm(
    make_llm: MakeLlm,
) -> None:
    llm, _ = make_llm(turns(tool_turn(("s1", "search_report", {"query": "revenue"})), "{}"))
    elsewhere = Usage(input_tokens=50_000, output_tokens=5_000)

    def search_while_another_job_bills(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        llm.budget.charge(GENERALIST_MODEL, elsewhere)  # a concurrent worker on the same llm
        return Served()(name, arguments)

    told = research(llm, search_while_another_job_bills, GENERALIST_MODEL)(JOB)

    other = cost_usd(GENERALIST_MODEL, 50_000, 5_000)
    assert told.usd == pytest.approx(llm.spent_usd - other), (told.usd, llm.spent_usd, other)
    assert 0 < told.usd < other


def test_a_fallback_asks_the_generalist_as_the_model_it_was_given(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    figures, _ = small("hostllm-analyst", completion("no figures"))
    generalist, sdk = make_llm(
        turns(tool_turn(("s1", "search_report", {"query": "revenue"})), '{"figures": []}')
    )
    route = router()

    route(report_analyst(figures, generalist, Served(), model="claude-haiku-4-5"), JOB)

    assert [logged.path for logged in route.routed] == ["generalist"]
    assert [sent.model for sent in sdk.sent] == ["claude-haiku-4-5"] * 2


def test_a_call_to_any_other_tool_is_refused_and_runs_nothing(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    reading = asking(("read_file", {"file_path": "notes.txt"}))
    llm, fake = small("hostllm-analyst", reading, completion('{"figures": []}'))
    served = Served()

    told = report_analyst(llm, big(make_llm, "unused"), served).ask(JOB)

    [result] = [
        m for m in cast(list[dict[str, object]], fake.sent[1]["messages"]) if m["role"] == "tool"
    ]
    assert result["content"] == "Error: read_file is not available here"
    assert told.calls == () and told.read == () and served.asked == []


def test_a_model_still_working_after_its_turns_has_no_answer_to_check(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, fake = small("hostllm-analyst", SEARCHING)

    told = report_analyst(llm, big(make_llm, "unused"), Served()).ask(JOB)

    assert told.text == "" and len(fake.sent) == SPECIALIST_TOOL_TURNS
    assert len(told.read) == SPECIALIST_TOOL_TURNS


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
    llm, fake = small("hostllm-analyst", dropped, completion('{"figures": []}'))

    told = report_analyst(llm, big(make_llm, "unused"), Served()).ask(JOB)

    assert len(fake.sent) == 2 and told.text == '{"figures": []}'


def test_an_endpoint_that_stays_down_is_unreachable_and_the_generalist_is_not_asked(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    llm, fake = small("hostllm-analyst", DROPPED)
    generalist, sdk = make_llm(FIGURES)
    route = router()

    with pytest.raises(SpecialistUnreachable, match="hostllm-analyst did not answer in 2 tries"):
        route(report_analyst(llm, generalist, Served()), JOB)
    assert len(fake.sent) == 2 and sdk.sent == []
    assert route.routed[0].path is None and "unreachable" in str(route.routed[0].failure)


def test_a_refusal_that_is_not_the_endpoint_being_down_is_not_retried(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    denied = openai.AuthenticationError(
        "bad proxy token", response=httpx.Response(401, request=REQUEST), body=None
    )
    llm, fake = small("hostllm-analyst", denied)

    with pytest.raises(openai.AuthenticationError):
        report_analyst(llm, big(make_llm, "unused"), Served()).ask(JOB)
    assert len(fake.sent) == 1


def test_the_warm_up_asks_the_endpoint_for_one_token(offline_tokenizer: None) -> None:
    llm, fake = small("hostllm-analyst", DROPPED, completion("ok", completion=1))

    warm_up(llm)

    assert [sent["max_tokens"] for sent in fake.sent] == [1, 1], "sent again after a drop"
    assert fake.sent[0]["model"] == "report-analyst" and llm.spent_usd > 0


def test_reaching_gives_up_after_its_tries(offline_tokenizer: None) -> None:
    llm, _ = small("hostllm-analyst", DROPPED)
    tried: list[int] = []

    def call() -> None:
        tried.append(1)
        raise DROPPED

    with pytest.raises(SpecialistUnreachable, match="hostllm-analyst"):
        reaching(llm, call)
    assert len(tried) == 2


def test_the_prompt_gives_the_turns_the_loop_allows() -> None:
    words = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    system, _ = research_messages("", "", "")

    assert f"You have {words[SPECIALIST_TOOL_TURNS]} turns" in system


def test_the_prompt_reads_a_labels_multiplier_into_the_calculate_call_the_check_accepts() -> None:
    system, _ = research_messages("", "", "")

    assert "'= ×' and that scale's multiplier" in system
    assert "calculate('1250 * 10000000')" in system
    assert canonical("1250 * 10000000") == "1250 * 10**7", "the check reads it as a power of ten"
