"""Project 2's agent as a LangGraph graph: `agent`'s prompt, tools and last answer, with the loop
drawn as nodes instead of left to the SDK's tool runner.

uv run --group graph python -m aigent.agent_graph ["question"] [--yes]

A `model` node sends one turn through `Llm.turn`, a `tools` node runs what it asked for, and a turn
that the cap or the budget refuses becomes one last answer through `Llm.last_turn`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aigent.agent import FINISH, QUESTIONS, SYSTEM, TOOLS, called
from aigent.config import MAX_AGENT_TURNS, MAX_USD_PER_RUN
from aigent.config import MAX_TOKENS_TOOL_LOOP as MAX_TOKENS
from aigent.errors import BudgetExceeded, TurnsExhausted
from aigent.llm import Dispatch, Llm, Request
from aigent.messages import Block, Msg, Reply
from aigent.report_tools import SEARCH_TOOL, ReportSearch, Searched
from aigent.tools import execute_tool
from aigent.workflows.demo import run_demo
from aigent.workflows.reports import Search


class State(TypedDict):
    messages: list[Msg]
    tools: list[str]  # every tool called, in order, the API's own included
    turns: int
    reply: Reply | None
    stopped: str | None


@dataclass(frozen=True)
class Traced:
    """How a run went: its answer, the tools it called in order, and why it stopped early."""

    answer: str | None
    tools: list[str]
    turns: int
    stopped: str | None
    searches: list[Searched]


def results(reply: Reply, dispatch: Dispatch) -> Msg:
    """Every tool call of a turn, run, and answered in one user message; a failure is an error
    result rather than an exception."""
    answered: list[Block] = []
    for block in reply.blocks:
        if block.get("type") != "tool_use":
            continue
        arguments = cast(dict[str, object], block.get("input") or {})
        content, is_error = dispatch(str(block["name"]), arguments)
        answered.append(
            {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": content,
                "is_error": is_error,
            }
        )
    return {"role": "user", "content": answered}


def build(
    llm: Llm, dispatch: Dispatch, max_turns: int = MAX_AGENT_TURNS, model: str | None = None
) -> CompiledStateGraph[State, None, State, State]:
    """The graph, compiled. Invoke it with a question's first user message. `model` overrides the
    one the client serves."""

    def upcoming(state: State) -> Request:
        return Request(
            f"agent:{state['turns'] + 1}",
            state["messages"],
            MAX_TOKENS,
            system=SYSTEM,
            model=model,
            tools=TOOLS,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral"},
        )

    def took(state: State, reply: Reply, stopped: str | None = None) -> dict[str, object]:
        return {
            "messages": [*state["messages"], {"role": "assistant", "content": reply.blocks}],
            "tools": [*state["tools"], *called(reply)],
            "turns": state["turns"] + 1,
            "reply": reply,
            "stopped": stopped,
        }

    def last(state: State, stop: Exception) -> dict[str, object]:
        try:
            return took(state, llm.last_turn(upcoming(state), FINISH, stop), str(stop))
        except (BudgetExceeded, TurnsExhausted) as refused:
            return {"reply": None, "stopped": str(refused)}

    def think(state: State) -> dict[str, object]:
        after_tools = state["turns"] > 0
        if after_tools and state["turns"] + 1 == max_turns:
            return last(
                state, TurnsExhausted(f"the last of {max_turns} turns is kept for an answer")
            )
        try:
            return took(state, llm.turn(upcoming(state)))
        except BudgetExceeded as refused:
            if not after_tools:
                return {"reply": None, "stopped": str(refused)}
            return last(state, refused)

    def tools(state: State) -> dict[str, object]:
        assert state["reply"] is not None
        return {"messages": [*state["messages"], results(state["reply"], dispatch)]}

    def route(state: State) -> str:
        reply = state["reply"]
        if reply is None or state["stopped"] is not None:
            return END
        if reply.stop_reason == "tool_use":
            return "tools"
        return "model" if reply.stop_reason == "pause_turn" else END

    graph = StateGraph(State)
    graph.add_node("model", think)
    graph.add_node("tools", tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", "model", END])
    graph.add_edge("tools", "model")
    return graph.compile()


def run(
    llm: Llm,
    search: Search,
    question: str,
    max_turns: int = MAX_AGENT_TURNS,
    model: str | None = None,
) -> Traced:
    reports = ReportSearch(search)

    def dispatch(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        if name == SEARCH_TOOL["name"]:
            return (reports(arguments), False)
        return execute_tool(name, arguments)

    start: State = {
        "messages": [{"role": "user", "content": question}],
        "tools": [],
        "turns": 0,
        "reply": None,
        "stopped": None,
    }
    end = cast(
        State,
        build(llm, dispatch, max_turns, model).invoke(
            start, {"recursion_limit": 2 * max_turns + 4}
        ),
    )
    reply = end["reply"]
    answer = reply.text if reply is not None else None
    return Traced(answer, end["tools"], end["turns"], end["stopped"], reports.searches)


def show(result: Traced) -> str:
    lines = [f"{result.turns} turns; tools called: {', '.join(result.tools) or 'none'}"]
    if result.stopped is not None:
        lines.append(f"\nsearching stopped: {result.stopped}")
    if result.answer is not None:
        lines.append(f"\n{result.answer}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    run_demo(
        argv,
        prog="aigent.agent_graph",
        questions=QUESTIONS,
        run=run,
        show=show,
        limit_usd=MAX_USD_PER_RUN,
        scope="run",
    )


if __name__ == "__main__":
    main()
