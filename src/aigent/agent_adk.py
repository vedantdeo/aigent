"""Project 2's agent on Google's ADK: `agent`'s prompt and tools, with ADK's runner driving the loop
and every model call still made through `llm`.

uv run --group adk python -m aigent.agent_adk ["question"] [--yes]

`ThroughLlm` is ADK's model extension point pointed at `Llm.turn`, so the per-turn and per-task
ceilings hold. ADK's Anthropic converter refuses server-side web-search blocks, so a turn's search
is followed inside the turn and only the blocks ADK can carry go back into its history.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any, cast

from anthropic.types import Message
from google.adk.agents import LlmAgent, RunConfig
from google.adk.models.anthropic_llm import (
    content_to_message_param,
    function_declaration_to_tool_param,
    message_to_generate_content_response,
)
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from pydantic import ConfigDict, PrivateAttr

from aigent.adapters import spec
from aigent.agent import FINISH, QUESTIONS, SYSTEM
from aigent.agent_graph import Traced, called
from aigent.config import MAX_AGENT_TURNS, MAX_USD_PER_RUN
from aigent.config import MAX_TOKENS_TOOL_LOOP as MAX_TOKENS
from aigent.errors import BudgetExceeded, TurnsExhausted
from aigent.llm import Llm, Request
from aigent.messages import Msg, Reply, Tool
from aigent.report_tools import ReportSearch
from aigent.tools import WEB_SEARCH_TOOL, execute_tool
from aigent.workflows.demo import run_demo
from aigent.workflows.reports import Search

KEPT = ("text", "thinking", "redacted_thinking", "tool_use")  # what ADK's converter can carry


class ThroughLlm(BaseLlm):
    """ADK's model, sending every turn through `Llm.turn` and recording the tools each asked for."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Llm
    max_turns: int = MAX_AGENT_TURNS
    _tools: list[str] = PrivateAttr(default_factory=list[str])
    _turns: int = PrivateAttr(default=0)
    _stopped: str | None = PrivateAttr(default=None)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        messages: list[Msg] = [
            cast(Msg, message)
            for message in (content_to_message_param(c) for c in llm_request.contents or [])
            if message["content"]
        ]
        configured = llm_request.config.tools if llm_request.config else None
        declared = [
            fd
            for tool in configured or []
            if isinstance(tool, types.Tool)
            for fd in tool.function_declarations or []
        ]
        tools: list[Tool] = [
            *(cast(Tool, function_declaration_to_tool_param(fd)) for fd in declared),
            cast(Tool, WEB_SEARCH_TOOL),
        ]
        reply = await asyncio.to_thread(self._answer, messages, tools)
        yield message_to_generate_content_response(_carryable(reply))

    def _request(self, messages: Sequence[Msg], tools: Sequence[Tool]) -> Request:
        return Request(
            f"agent:{self._turns + 1}",
            messages,
            MAX_TOKENS,
            system=SYSTEM,
            model=self.model,
            tools=tools,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral"},
        )

    def _answer(self, messages: list[Msg], tools: list[Tool]) -> Reply:
        """One turn, following a paused server-side search to its end; the last turn under the
        cap, or one the budget refuses, is sent as a last answer instead."""
        after_tools = self._turns > 0
        upcoming = self._request(messages, tools)
        if after_tools and self._turns + 1 >= self.max_turns:
            return self._last(upcoming, TurnsExhausted(f"the last of {self.max_turns} turns"))
        try:
            reply = self.llm.turn(upcoming)
        except BudgetExceeded as refused:
            if not after_tools:
                raise
            return self._last(upcoming, refused)
        self._took(reply)
        while reply.stop_reason == "pause_turn":
            messages = [*messages, {"role": "assistant", "content": reply.blocks}]
            reply = self.llm.turn(self._request(messages, tools))
            self._took(reply)
        return reply

    def _last(self, upcoming: Request, stop: Exception) -> Reply:
        self._stopped = str(stop)
        reply = self.llm.last_turn(upcoming, FINISH, stop)
        self._took(reply)
        return reply

    def _took(self, reply: Reply) -> None:
        self._turns += 1
        self._tools.extend(called(reply))


def _carryable(reply: Reply) -> Message:
    """The reply with only the blocks ADK's converter accepts; server-side search is dropped."""
    message = cast(Message, reply.raw)
    kept = [block for block in message.content if block.type in KEPT]
    return message.model_copy(update={"content": kept})


def tools_for(search: Search) -> tuple[list[FunctionTool], ReportSearch]:
    """Our tools as ADK functions: ADK derives each schema from the signature and docstring."""
    reports = ReportSearch(search)

    def search_reports(query: str, report: str) -> str:
        """Search the three FY25 annual reports. report is ITC-FY25, RELIANCE-FY25, TATAMOTORS-FY25
        or all. Different words find different passages, so rephrase when a search misses."""
        return reports({"query": query, "report": report})

    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression. Use this instead of computing in your head."""
        return execute_tool("calculate", {"expression": expression})[0]

    def current_time() -> str:
        """The current date and time in UTC."""
        return execute_tool("current_time", {})[0]

    def read_file(file_path: str) -> str:
        """Read a text file from the sandbox by its path."""
        return execute_tool("read_file", {"file_path": file_path})[0]

    functions = (search_reports, calculate, current_time, read_file)
    return [FunctionTool(f) for f in functions], reports


def run(
    llm: Llm,
    search: Search,
    question: str,
    max_turns: int = MAX_AGENT_TURNS,
    model: str | None = None,
) -> Traced:
    tools, reports = tools_for(search)
    brain = ThroughLlm(model=model or spec(llm.client).model, llm=llm, max_turns=max_turns)
    agent = LlmAgent(name="aigent", model=brain, instruction=SYSTEM, tools=cast(list[Any], tools))
    runner = InMemoryRunner(agent=agent, app_name="aigent")
    session = asyncio.run(runner.session_service.create_session(app_name="aigent", user_id="u"))
    asked = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    answer: str | None = None
    stopped: str | None = None
    try:
        for event in runner.run(
            user_id="u",
            session_id=session.id,
            new_message=asked,
            run_config=RunConfig(max_llm_calls=max_turns + 1),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                answer = "".join(p.text or "" for p in event.content.parts if not p.thought)
    except (BudgetExceeded, TurnsExhausted) as refused:
        stopped = str(refused)
    return Traced(
        answer or None,
        list(brain._tools),
        brain._turns,
        stopped or brain._stopped,
        reports.searches,
    )


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
        prog="aigent.agent_adk",
        questions=QUESTIONS,
        run=run,
        show=show,
        limit_usd=MAX_USD_PER_RUN,
        scope="run",
    )


if __name__ == "__main__":
    main()
