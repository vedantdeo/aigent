"""Orchestrator-workers again, as a LangGraph graph: the same prompts, plan and calls, as nodes.

uv run --group graph python -m aigent.workflows.orchestrator_workers_graph ["question"] [--yes]

Only the control flow differs from `orchestrator_workers`, so the two can be compared line for line.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from aigent.config import (
    MAX_GRAPH_STEPS,
    MAX_PLAN_TASKS,
    MAX_TOKENS_PLAN,
    MAX_TOKENS_SYNTHESIS,
    MAX_TOKENS_WORKER,
)
from aigent.config import THINKING_WORKFLOW_PARAM as THINKING
from aigent.llm.core import Llm, Request
from aigent.retrieval.chunk import context_block
from aigent.workflows.orchestrator_workers import (
    PLAN,
    SYNTHESISE,
    WORK,
    Orchestrated,
    Plan,
    Subtask,
)
from aigent.workflows.reports import Search


class State(TypedDict):
    question: str
    plan: list[Subtask]
    passages: list[str]
    # Workers write in parallel, so this key needs a reducer; the number restores plan order.
    findings: Annotated[list[tuple[int, str]], operator.add]
    answer: str


class Brief(TypedDict):
    number: int
    task: Subtask
    passages: str


def build(llm: Llm, search: Search) -> CompiledStateGraph[State, None, State, State]:
    """The graph, compiled: to run, stream, draw, or checkpoint."""

    def plan(state: State) -> dict[str, object]:
        planning = Request.ask("plan", PLAN, state["question"], MAX_TOKENS_PLAN, thinking=THINKING)
        return {"plan": llm.record(planning, Plan).subtasks[:MAX_PLAN_TASKS]}

    def gather(state: State) -> dict[str, object]:
        # One search at a time, here: the local models are not shared across threads.
        return {
            "passages": [context_block(search(t.query, doc_id=t.report)) for t in state["plan"]]
        }

    def fan_out(state: State) -> list[Send]:
        briefs = enumerate(zip(state["plan"], state["passages"], strict=True), 1)
        return [Send("work", Brief(number=n, task=t, passages=p)) for n, (t, p) in briefs]

    def work(state: Brief) -> dict[str, object]:
        prompt = f"{state['passages']}\n\nbrief: {state['task'].brief}"
        asked = Request.ask(
            f"worker:{state['number']}", WORK, prompt, MAX_TOKENS_WORKER, thinking=THINKING
        )
        return {"findings": [(state["number"], llm.text(asked))]}

    def synthesise(state: State) -> dict[str, object]:
        listed = "\n\n".join(
            f"<finding brief={task.brief!r}>\n{found}\n</finding>"
            for task, (_, found) in zip(state["plan"], sorted(state["findings"]), strict=True)
        )
        prompt = f"question: {state['question']}\n\n{listed}"
        last = Request.ask(
            "synthesise", SYNTHESISE, prompt, MAX_TOKENS_SYNTHESIS, thinking=THINKING
        )
        return {"answer": llm.text(last)}

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("gather", gather)
    graph.add_node("work", work, input_schema=Brief)
    graph.add_node("synthesise", synthesise)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", lambda state: "gather" if state["plan"] else END, ["gather", END]
    )
    graph.add_conditional_edges("gather", fan_out, ["work"])
    graph.add_edge("work", "synthesise")
    graph.add_edge("synthesise", END)
    return graph.compile()


def run(llm: Llm, search: Search, question: str) -> Orchestrated:
    start = State(question=question, plan=[], passages=[], findings=[], answer="")
    graph = build(llm, search)
    final = cast(State, graph.invoke(start, {"recursion_limit": MAX_GRAPH_STEPS}))
    if not final["plan"]:
        return Orchestrated([], [], "(the orchestrator planned no subtasks)")
    findings = [found for _, found in sorted(final["findings"])]
    return Orchestrated(final["plan"], findings, final["answer"])
