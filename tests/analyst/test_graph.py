"""The capstone's graph over scripted models: research, the memo, the critic's loop and its cap,
and the gate, by an approver and by a person resuming a paused run."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from anthropic.types import Message, TextBlock
from pydantic import BaseModel

pytest.importorskip("langgraph")

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from aigent.analyst import graph as graph_module  # noqa: E402
from aigent.analyst.experts import EXTRACT_FIGURES_TOOL, READ_HEADLINE_TOOL  # noqa: E402
from aigent.analyst.graph import RESEARCH, State, analysed, begin, build, run  # noqa: E402
from aigent.analyst.memo import Claim, Judged, Memo, Review  # noqa: E402
from aigent.config import (  # noqa: E402
    CACHE_PARAM,
    CRITIC_MODEL,
    GENERALIST_MODEL,
    GENERALIST_THINKING,
    MAX_CRITIC_REVISIONS,
)
from aigent.errors import GuardrailTripped  # noqa: E402
from aigent.llm.adapters.cfg_local import CLIENT as LOCAL  # noqa: E402
from aigent.llm.adapters.cfg_sarvam import THOUGHT  # noqa: E402
from aigent.llm.core import Llm  # noqa: E402

from ..conftest import (  # noqa: E402
    PASSAGES,
    FakeOpenAI,
    MakeLlm,
    Sent,
    completion,
    tool_turn,
)
from .conftest import DIVIDEND, box, search  # noqa: E402

QUESTION = "What dividend did Reliance recommend for FY25?"
SEARCHING = tool_turn(("s1", "search_filings", {"query": "dividend", "company": "RELIANCE"}))
DRAFT = f"Reliance recommended Rs 5.50 a share [{DIVIDEND}]."
PASSAGE = PASSAGES["RELIANCE-FY25"][0].text
OTHER = PASSAGES["ITC-FY25"][0].text
RIGHT = Claim(id="c1", text="Reliance recommended Rs 5.50 a share for FY25.", citations=[DIVIDEND])
WRONG = Claim(id="c1", text="Reliance recommended Rs 6.50 a share for FY25.", citations=[DIVIDEND])
SUPPORTED = Judged(verdict="supported", reason="the passage says Rs 5.50")
REFUSED = Judged(verdict="wrong", reason="the passage says Rs 5.50, not 6.50")


def memo(*claims: Claim) -> Memo:
    return Memo(summary="Reliance recommended Rs 5.50.", claims=list(claims), caveats=[])


def generalist(*memos: Memo | None) -> Callable[[Sent], Message | str | BaseModel | None]:
    """Each research pass searches once, then drafts; each memo request gets the next memo."""
    written: list[int] = []

    def reply(sent: Sent) -> Message | str | BaseModel | None:
        if sent.schema is Memo:
            written.append(1)
            return memos[min(len(written), len(memos)) - 1]
        return SEARCHING if len(sent.messages) == 1 else DRAFT

    return reply


def judging(sent: Sent) -> Judged:
    return SUPPORTED if "5.50 a share" in str(sent.messages[0]["content"]) else REFUSED


def asked(sent: Sent) -> str:
    return str(sent.messages[0]["content"])


def test_a_supported_memo_reaches_the_gate_and_is_approved(make_llm: MakeLlm) -> None:
    writer, sent = make_llm(generalist(memo(RIGHT)))
    critic, judged = make_llm(judging)
    tools = box(search)

    done = run(writer, critic, tools, f"  {QUESTION}\n")

    assert done.memo == memo(RIGHT) and done.verdicts == {"c1": SUPPORTED}
    assert done.review == Review(approved=True) and done.revisions == 0 and done.stopped is None
    assert done.draft == DRAFT, "the generalist's own words, where it explains its choices"
    research, _, writing = sent.sent
    assert asked(research) == f"<question>{QUESTION}</question>", "trimmed by the input guard"
    assert [tool["name"] for tool in cast(list[dict[str, object]], research.tools)] == [
        "search_filings",
        "read_headline",
        "extract_figures",
    ]
    assert writing.schema is Memo and DRAFT in asked(writing) and DIVIDEND in asked(writing)
    assert writing.model == research.model == GENERALIST_MODEL
    assert research.thinking == GENERALIST_THINKING and research.cache_control == CACHE_PARAM


def test_the_generalist_is_asked_as_the_model_and_the_thinking_it_is_given(
    make_llm: MakeLlm,
) -> None:
    writer, sent = make_llm(generalist(memo(RIGHT)))

    run(writer, make_llm(judging)[0], box(search), QUESTION, "claude-haiku-4-5", None)

    research, _, writing = sent.sent
    assert writing.model == research.model == "claude-haiku-4-5"
    assert not research.thinking, research.thinking


def test_a_generalist_on_a_wire_with_no_cache_runs_as_its_client_s_own_model(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    fake = FakeOpenAI(completion(DRAFT), completion(memo(RIGHT).model_dump_json()))
    writer = Llm("local", sdk=cast(object, fake))
    tools = box(search)
    tools("search_filings", {})  # the passage the memo cites, already retrieved

    done = run(writer, make_llm(judging)[0], tools, QUESTION, None, None)

    assert done.memo == memo(RIGHT) and done.review == Review(approved=True)
    assert [sent["model"] for sent in fake.sent] == [LOCAL.model] * 2


def test_each_model_is_capped_as_its_own_client_says(
    make_llm: MakeLlm, offline_tokenizer: None
) -> None:
    fake = FakeOpenAI(completion(DRAFT), completion(memo(RIGHT).model_dump_json()))
    writer = Llm("sarvam-thinking", sdk=cast(object, fake))
    tools = box(search)
    tools("search_filings", {})

    run(writer, make_llm(judging)[0], tools, QUESTION, None, None)

    assert [sent["max_tokens"] for sent in fake.sent] == [THOUGHT["TOOL_LOOP"], THOUGHT["MEMO"]]
    assert all(sent["extra_body"] == {"reasoning_effort": "high"} for sent in fake.sent)


def test_the_draft_is_everything_the_generalist_said_not_only_its_last_turn(
    make_llm: MakeLlm,
) -> None:
    answering = SEARCHING.model_copy(
        update={"content": [TextBlock(type="text", text=DRAFT), *SEARCHING.content]}
    )
    closing = "Specialists: none, the passage prints the figure in full."

    def reply(sent: Sent) -> Message | str | BaseModel | None:
        if sent.schema is Memo:
            return memo(RIGHT)
        return answering if len(sent.messages) == 1 else closing

    writer, sent = make_llm(reply)

    done = run(writer, make_llm(judging)[0], box(search), QUESTION)

    assert done.draft == f"{DRAFT}\n\n{closing}"
    assert DRAFT in asked(sent.sent[-1]), "the writer is given the answer, not just its last line"


def test_the_research_prompt_names_no_specialist_so_a_new_one_needs_only_its_card() -> None:
    cards = [READ_HEADLINE_TOOL, EXTRACT_FIGURES_TOOL]

    assert "specialist" in RESEARCH and not [c["name"] for c in cards if c["name"] in RESEARCH]
    assert "Specialists:" in RESEARCH and "explain that decision" in RESEARCH


def test_the_critic_is_another_model_shown_only_the_claim_and_what_it_cites(
    make_llm: MakeLlm, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert CRITIC_MODEL != GENERALIST_MODEL
    monkeypatch.setattr(graph_module, "CRITIC_MODEL", "claude-haiku-4-5")  # not a client default
    writer, _ = make_llm(generalist(memo(RIGHT)))
    critic, judged = make_llm(judging)

    run(writer, critic, box(search), QUESTION)

    [request] = judged.sent
    assert request.model == "claude-haiku-4-5" and request.schema is Judged
    assert RIGHT.text in asked(request) and PASSAGE in asked(request)
    assert OTHER not in asked(request), "a passage the claim does not cite"
    assert DRAFT not in asked(request) and QUESTION not in asked(request)


def test_a_claim_the_critic_refuses_is_sent_back_with_its_reason_and_its_sources(
    make_llm: MakeLlm,
) -> None:
    writer, sent = make_llm(generalist(memo(WRONG), memo(RIGHT)))
    critic, _ = make_llm(judging)

    done = run(writer, critic, box(search), QUESTION)

    assert done.memo == memo(RIGHT) and done.verdicts == {"c1": SUPPORTED} and done.revisions == 1
    again = asked(sent.sent[3])
    assert f"<question>{QUESTION}</question>" in again and WRONG.text in again
    assert 'verdict="wrong"' in again and REFUSED.reason in again and PASSAGE in again


def test_the_critic_s_loop_ends_at_its_cap_and_a_person_sees_what_is_still_refused(
    make_llm: MakeLlm,
) -> None:
    writer, sent = make_llm(generalist(memo(WRONG)))
    critic, judged = make_llm(judging)
    shown: list[dict[str, Judged]] = []

    def approve(written: Memo, verdicts: dict[str, Judged]) -> Review:
        shown.append(verdicts)
        return Review(approved=True)

    graph = build(writer, critic, box(search), approve=approve)
    state = cast(State, graph.invoke(begin(QUESTION)))

    assert state["revisions"] == MAX_CRITIC_REVISIONS == 2 and shown == [{"c1": REFUSED}]
    assert len([s for s in sent.sent if s.schema is Memo]) == 3, "the draft and two redrafts"
    assert len(judged.sent) == 3


def test_a_claim_citing_something_never_retrieved_is_refused_without_asking_the_critic(
    make_llm: MakeLlm,
) -> None:
    invented = Claim(id="c1", text="ITC paid 14.35 a share.", citations=["ITC-FY25#0999"])
    writer, _ = make_llm(generalist(memo(invented), memo(RIGHT)))
    critic, judged = make_llm(judging)

    done = run(writer, critic, box(search), QUESTION)

    assert done.revisions == 1 and done.verdicts == {"c1": SUPPORTED}
    assert len(judged.sent) == 1, "only the redraft's claim reached the critic"


def test_a_critic_that_returns_no_verdict_counts_as_unsupported(make_llm: MakeLlm) -> None:
    writer, _ = make_llm(generalist(memo(RIGHT)))
    critic, _ = make_llm(lambda sent: None)

    done = run(writer, critic, box(search), QUESTION)

    assert done.verdicts["c1"] == Judged(
        verdict="unsupported", reason="the critic returned no verdict (it stopped on end_turn)"
    )
    assert done.revisions == MAX_CRITIC_REVISIONS


def test_an_answer_that_cannot_be_read_as_a_memo_stops_the_run(make_llm: MakeLlm) -> None:
    writer, _ = make_llm(generalist(None))
    critic, judged = make_llm(judging)

    done = run(writer, critic, box(search), QUESTION)

    assert done.memo is None and done.review is None and "could not be read" in str(done.stopped)
    assert judged.sent == []


def test_research_with_no_room_even_to_answer_stops_the_run(make_llm: MakeLlm) -> None:
    writer, sent = make_llm(generalist(memo(RIGHT)), limit_usd=0.0001)
    critic, _ = make_llm(judging)

    done = run(writer, critic, box(search), QUESTION)

    assert done.memo is None and done.stopped is not None and sent.sent == []


def test_a_question_the_input_guard_refuses_starts_nothing(make_llm: MakeLlm) -> None:
    writer, sent = make_llm(generalist(memo(RIGHT)))

    with pytest.raises(GuardrailTripped):
        run(writer, make_llm(judging)[0], box(search), "   ")
    assert sent.sent == []


def test_a_person_s_rejection_becomes_a_constraint_and_their_approval_ends_the_run(
    make_llm: MakeLlm,
) -> None:
    writer, sent = make_llm(generalist(memo(RIGHT)))
    critic, _ = make_llm(judging)
    tools = box(search)
    graph = build(writer, critic, tools, checkpointer=InMemorySaver())
    thread: RunnableConfig = {"configurable": {"thread_id": "t1"}}

    paused = graph.invoke(begin(QUESTION), thread)
    [waiting] = paused["__interrupt__"]
    assert waiting.value["memo"] == memo(RIGHT).model_dump()
    assert waiting.value["verdicts"] == {"c1": SUPPORTED.model_dump()}

    note = "say which board meeting recommended it"
    paused = graph.invoke(Command(resume={"approved": False, "note": note}), thread)
    assert "__interrupt__" in paused, "redrafted, judged, and back at the gate"
    assert f"<reviewer>{note}</reviewer>" in asked(sent.sent[3])

    state = cast(State, graph.invoke(Command(resume={"approved": True}), thread))
    done = analysed(state, tools)
    assert done.review == Review(approved=True) and done.revisions == 0
    assert state["notes"] == [note]
