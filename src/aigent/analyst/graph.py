"""The capstone's graph: the generalist researches with its toolbox and writes a memo of cited
claims, a critic on another model checks each claim against only what it cites, and a person
approves the memo or sends it back with a note.

    research → write → criticise → (revise → research)* → gate → END | research
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from aigent import guardrails, tracing
from aigent.analyst.cfg_analyst import (
    CRITIC_MODEL,
    GENERALIST_CALM,
    GENERALIST_MODEL,
    GENERALIST_THINKING,
    MAX_CRITIC_REVISIONS,
    max_tokens,
)
from aigent.analyst.memo import Judged, Memo, Review
from aigent.analyst.router import Routing
from aigent.analyst.toolbox import Toolbox
from aigent.config import CACHE_PARAM, MAX_AGENT_TURNS
from aigent.errors import BudgetExceeded, TurnsExhausted
from aigent.llm.adapters import spec
from aigent.llm.core import Llm, Request
from aigent.llm.messages import Thinking

RESEARCH = (
    "You are a filings analyst answering a question about Indian listed companies from their "
    "annual reports. Find what you need with search_filings, scoped to the company and the "
    "financial year asked about, and with fundamentals for figures in a company's results "
    "filings, which it returns exactly. "
    "Some of your tools are specialists: narrow models, each built for one job, which its "
    "tool's description states. Before you do a piece of work yourself, read what every "
    "specialist at your disposal is for and hand the work to the most applicable one; break a "
    "task into several specialist calls when it needs more than one. Use calculate for every "
    "piece of arithmetic. When you "
    "have enough, answer in prose: state each fact with its figure and year, and after it the id "
    "of every source it rests on in square brackets, a passage id or a fundamentals cite_as. Say "
    "plainly what the sources do not cover. End your answer with one line that begins "
    "'Specialists:' and names each specialist you called and for what; if you decided not to "
    "call any, explain that decision there, specialist by specialist. Only a tool whose "
    f"description begins 'Specialist for' is a specialist. {guardrails.UNTRUSTED_NOTE}"
)
FINISH = (
    "You are out of turns: do not call any more tools. Answer now from what you have, cite it, "
    "and say plainly what you could not check."
)
WRITE = (
    "Turn an analyst's answer into a memo. Keep only facts the answer states with a source id in "
    "square brackets; each becomes one claim citing exactly those ids. Add nothing the answer "
    "does not say. A caveat is something the sources did not cover or could not settle: never a "
    "remark about the answer itself, its method or an earlier draft. Leave out the answer's "
    "closing 'Specialists:' line and anything else it says about the tools it used. The memo "
    "is a JSON object with three keys: summary, two sentences answering the question; claims, a "
    "list of objects, each with an id (c1, c2, … in order), a text (one sentence stating one "
    "fact, with its figure and year) and citations (a list of the ids the fact rests on, at "
    "least one); and caveats, a list of sentences."
)
CRITIC = (
    "You check one claim from an analyst's memo against the sources it cites, and nothing else: "
    "you have not seen the analyst's reasoning and must not supply facts of your own. Decide "
    "whether the sources state the claim, do not state it, or contradict it, minding the figure, "
    "its unit and scale, the financial year and the company. Keep the reason to one sentence and "
    "quote at most a dozen words of a source. "
    f"{guardrails.UNTRUSTED_NOTE}"
)

Approve = Callable[[Memo, dict[str, Judged]], Review]


class State(TypedDict):
    question: str
    draft: str | None  # the generalist's answer in prose, citing ids
    memo: Memo | None
    verdicts: dict[str, Judged]  # by claim id
    revisions: int  # critic loops used
    notes: list[str]  # a person's rejections, each a constraint from then on
    review: Review | None
    stopped: str | None


@dataclass(frozen=True)
class Analysed:
    """How a run ended: the memo and the critic's verdicts on it, a person's decision if it got
    that far, every specialist call, why it stopped early if it did, and the generalist's last
    answer in its own words, which is where it explains calling no specialist."""

    memo: Memo | None
    verdicts: dict[str, Judged]
    review: Review | None
    revisions: int
    routed: list[Routing] = field(default_factory=list[Routing])
    stopped: str | None = None
    draft: str | None = None


def unsupported(state: State) -> dict[str, Judged]:
    """The verdicts that are not `supported`, by claim id."""
    return {k: v for k, v in state["verdicts"].items() if v.verdict != "supported"}


def _sources(toolbox: Toolbox, citations: list[str]) -> str:
    return "\n\n".join(
        f'<source id="{c}">\n{toolbox.evidence[c].text}\n</source>'
        for c in citations
        if c in toolbox.evidence
    )


def _brief(state: State, toolbox: Toolbox) -> str:
    """What a research pass is asked: the question, then on a redraft the memo so far, what the
    critic refused with the sources it was judged on, and every note a person has left."""
    parts = [f"<question>{state['question']}</question>"]
    memo, refused = state["memo"], unsupported(state)
    if memo is not None and (refused or state["notes"]):
        parts.append(f"<memo>{memo.model_dump_json()}</memo>")
    for claim in memo.claims if memo is not None else []:
        if claim.id in refused:
            judged = refused[claim.id]
            parts.append(
                f'<refused claim="{claim.id}" verdict="{judged.verdict}">\n{judged.reason}\n'
                f"{_sources(toolbox, claim.citations)}\n</refused>"
            )
    parts += [f"<reviewer>{note}</reviewer>" for note in state["notes"]]
    if len(parts) > 1:
        parts.append(
            "Research again where needed and answer afresh, fixing what was refused. Write the "
            "answer as if for the first time: do not mention the earlier memo or what was wrong "
            "with it."
        )
    return "\n\n".join(parts)


def build(
    generalist: Llm,
    critic: Llm,
    toolbox: Toolbox,
    approve: Approve | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    max_turns: int = MAX_AGENT_TURNS,
    model: str | None = GENERALIST_MODEL,
    thinking: Thinking | None = GENERALIST_THINKING,
    calm: bool = False,
) -> CompiledStateGraph[State, None, State, State]:
    """The graph, compiled. With `approve`, the gate asks it; without, the gate interrupts and
    waits to be resumed with `{"approved": …, "note": …}`, which needs a `checkpointer`. The
    generalist is asked as `model`, or its client's own if None, thinking as `thinking` says; with
    `calm`, it stops thinking once a specialist has answered, for the rest of research, and never
    thinks writing the memo."""
    dispatch = tracing.traced_tools(toolbox, {"search_filings": "retriever"})
    memo_cap = max_tokens("MEMO", spec(generalist.client))
    verdict_cap = max_tokens("VERDICT", spec(critic.client))

    def answered() -> bool:
        return any(routed.path == "specialist" for routed in toolbox.route.routed)

    def thinking_now() -> Thinking | None:
        return GENERALIST_CALM if calm and answered() else thinking

    def research(state: State) -> dict[str, object]:
        request = Request(
            "research",
            [{"role": "user", "content": _brief(state, toolbox)}],
            max_tokens("TOOL_LOOP", spec(generalist.client)),
            system=RESEARCH,
            model=model,
            tools=toolbox.tools,
            thinking=thinking_now(),
            cache_control=CACHE_PARAM if "cache" in generalist.supports else None,
        )
        try:
            ran = generalist.run_tools(
                request,
                dispatch,
                max_turns=max_turns,
                finish=FINISH,
                rethink=thinking_now if calm else None,
            )
        except (BudgetExceeded, TurnsExhausted) as stop:
            return {"draft": None, "stopped": str(stop)}
        # every turn's text, not the last turn's: an answer is often written a turn before the end
        said = [turn.text.strip() for turn in ran.turns if turn.text.strip()]
        if not said:
            return {"draft": None, "stopped": "the generalist's answer was empty"}
        return {"draft": "\n\n".join(said), "stopped": ran.cut_short}

    def write(state: State) -> dict[str, object]:
        prompt = f"<question>{state['question']}</question>\n\n<answer>{state['draft']}</answer>"
        request = Request.ask("memo", WRITE, prompt, memo_cap, model=model)
        if calm:  # writing a memo from a draft is formatting: no reasoning to spend on it
            request = replace(request, thinking=GENERALIST_CALM)
        parsed = generalist.parse(request, Memo)
        if parsed.parsed is None:
            return {"memo": None, "stopped": "the generalist's answer could not be read as a memo"}
        if not parsed.parsed.claims:
            return {"memo": None, "stopped": "the memo has no claims"}
        return {"memo": parsed.parsed, "verdicts": {}}

    def criticise(state: State) -> dict[str, object]:
        memo = state["memo"]
        assert memo is not None
        verdicts: dict[str, Judged] = {}
        for claim in memo.claims:
            missing = [c for c in claim.citations if c not in toolbox.evidence]
            if missing:
                verdicts[claim.id] = Judged(
                    verdict="unsupported",
                    reason=f"it cites {', '.join(missing)}, which this run never retrieved",
                )
                continue
            prompt = f"<claim>{claim.text}</claim>\n\n{_sources(toolbox, claim.citations)}"
            request = Request.ask(
                f"critic:{claim.id}", CRITIC, prompt, verdict_cap, model=CRITIC_MODEL
            )
            told = critic.parse(request, Judged)
            verdicts[claim.id] = told.parsed or Judged(
                verdict="unsupported",
                reason=f"the critic returned no verdict (it stopped on {told.stop_reason})",
            )
        return {"verdicts": verdicts}

    def revise(state: State) -> dict[str, object]:
        return {"revisions": state["revisions"] + 1}

    def gate(state: State) -> dict[str, object]:
        memo = state["memo"]
        assert memo is not None
        if approve is not None:
            review = approve(memo, state["verdicts"])
        else:
            shown = {k: v.model_dump() for k, v in state["verdicts"].items()}
            review = Review.model_validate(
                interrupt({"memo": memo.model_dump(), "verdicts": shown})
            )
        notes = (
            state["notes"] if review.approved or not review.note else [*state["notes"], review.note]
        )
        return {"review": review, "notes": notes}

    def after_research(state: State) -> str:
        return END if state["draft"] is None else "write"

    def after_write(state: State) -> str:
        return END if state["memo"] is None else "criticise"

    def after_critic(state: State) -> str:
        if unsupported(state) and state["revisions"] < MAX_CRITIC_REVISIONS:
            return "revise"
        return "gate"

    def after_gate(state: State) -> str:
        review = state["review"]
        return END if review is None or review.approved else "research"

    graph = StateGraph(State)
    for name, node in (
        ("research", research),
        ("write", write),
        ("criticise", criticise),
        ("revise", revise),
        ("gate", gate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", after_research, ["write", END])
    graph.add_conditional_edges("write", after_write, ["criticise", END])
    graph.add_conditional_edges("criticise", after_critic, ["revise", "gate"])
    graph.add_edge("revise", "research")
    graph.add_conditional_edges("gate", after_gate, ["research", END])
    return graph.compile(checkpointer=checkpointer)


def begin(question: str) -> State:
    """The state a run starts from; the question goes through the input guard first."""
    return {
        "question": guardrails.check_task(question),
        "draft": None,
        "memo": None,
        "verdicts": {},
        "revisions": 0,
        "notes": [],
        "review": None,
        "stopped": None,
    }


def analysed(state: State, toolbox: Toolbox) -> Analysed:
    """A finished, or interrupted, run's state as its result."""
    return Analysed(
        state["memo"],
        state["verdicts"],
        state["review"],
        state["revisions"],
        list(toolbox.route.routed),
        state["stopped"],
        state["draft"],
    )


def always_approve(memo: Memo, verdicts: dict[str, Judged]) -> Review:
    """The evals' approver: the memo as the critic left it."""
    return Review(approved=True)


def run(
    generalist: Llm,
    critic: Llm,
    toolbox: Toolbox,
    question: str,
    model: str | None = GENERALIST_MODEL,
    thinking: Thinking | None = GENERALIST_THINKING,
    calm: bool = False,
) -> Analysed:
    """One question through the graph with no one at the gate, as an eval runs it."""
    graph = build(
        generalist,
        critic,
        toolbox,
        approve=always_approve,
        model=model,
        thinking=thinking,
        calm=calm,
    )
    return analysed(cast(State, graph.invoke(begin(question))), toolbox)
