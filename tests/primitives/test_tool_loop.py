"""The hand-written agent loop, tested two ways.

The unit tests script the model's responses with a fake client, so they check the loop's rules
(assistant turn echoed back, all tool results in one message, errors flagged not raised, turn cap)
for free. The live test at the bottom runs the real demo task and spends about two cents; it only
runs with `uv run pytest -m live`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import anthropic
import pytest
from anthropic.types import (
    ContentBlock,
    Message,
    MessageParam,
    MessageTokensCount,
    TextBlock,
    ToolResultBlockParam,
    ToolUseBlock,
    Usage,
)

from entropic.config import MAX_AGENT_TURNS as MAX_TURNS
from entropic.config import has_credentials
from entropic.errors import BudgetExceeded
from entropic.primitives.tool_loop import run

DEMO_TASK = (
    "If I invest 250,000 rupees today at 11.5% compounded annually, what is it worth after "
    "7 years? Also tell me the current UTC time, and how many hours until midnight UTC."
)


def _message(content: list[ContentBlock], stop_reason: str) -> Message:
    return Message(
        id="msg_fake",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=content,
        stop_reason=cast("anthropic.types.StopReason", stop_reason),
        stop_sequence=None,
        usage=Usage(input_tokens=100, output_tokens=50),
    )


def _tool_use(tool_id: str, name: str, **tool_input: object) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=tool_id, name=name, input=tool_input)


class _FakeMessages:
    """Stands in for client.messages: hands out scripted responses and records what was sent."""

    def __init__(self, responses: Sequence[Message]) -> None:
        self._responses = iter(responses)
        self.sent: list[list[MessageParam]] = []

    def count_tokens(self, **_: object) -> MessageTokensCount:
        return MessageTokensCount(input_tokens=100)

    def create(self, **kwargs: object) -> Message:
        self.sent.append(list(cast(Sequence[MessageParam], kwargs["messages"])))
        return next(self._responses)


class _FakeClient:
    def __init__(self, responses: Sequence[Message]) -> None:
        self.messages = _FakeMessages(responses)


def _client(responses: Sequence[Message]) -> tuple[anthropic.Anthropic, _FakeMessages]:
    fake = _FakeClient(responses)
    return cast(anthropic.Anthropic, fake), fake.messages


def _tool_results(message: MessageParam) -> list[ToolResultBlockParam]:
    """Narrow a user message's content to tool-result blocks, failing loudly on anything else."""
    content = message["content"]
    assert not isinstance(content, str), "tool results must be a list of blocks, not a string"
    results: list[ToolResultBlockParam] = []
    for block in content:
        assert isinstance(block, dict) and block.get("type") == "tool_result", block
        results.append(cast(ToolResultBlockParam, block))
    return results


def test_parallel_tool_results_go_back_in_one_message_and_errors_are_flagged() -> None:
    client, log = _client(
        [
            _message(
                [
                    TextBlock(type="text", text="Two calculations."),
                    _tool_use("toolu_1", "calculate", expression="2 ** 10"),
                    _tool_use("toolu_2", "calculate", expression="1 / 0"),
                ],
                "tool_use",
            ),
            _message([TextBlock(type="text", text="1024; the second one failed.")], "end_turn"),
        ]
    )

    answer = run("compute two things", client=client)

    assert answer == "1024; the second one failed."
    assert len(log.sent) == 2
    turn_two = log.sent[1]
    assert [m["role"] for m in turn_two] == ["user", "assistant", "user"]

    results = _tool_results(turn_two[2])
    assert len(results) == 2, "both tool results must travel in one user message"
    first, second = results
    assert first["tool_use_id"] == "toolu_1" and first.get("content") == "1024.0"
    assert "is_error" not in first
    assert second["tool_use_id"] == "toolu_2"
    assert second.get("is_error") is True
    assert str(second.get("content")).startswith("Error:")


def test_turn_cap_stops_a_loop_that_never_finishes() -> None:
    endless = _message([_tool_use("toolu_x", "current_time")], "tool_use")
    client, log = _client([endless] * (MAX_TURNS + 1))

    with pytest.raises(RuntimeError, match="did not finish"):
        run("loop forever", client=client)
    assert len(log.sent) == MAX_TURNS


def test_a_call_that_could_cross_the_run_ceiling_is_never_sent() -> None:
    """A turn's worst case is its input plus the full 4096-token cap, about $0.10 on Opus, so a
    $0.05 run cannot afford even the first one — and finds that out before sending it."""
    client, log = _client([_message([TextBlock(type="text", text="never sent")], "end_turn")])

    with pytest.raises(BudgetExceeded, match="per-run ceiling"):
        run("anything", client=client, limit_usd=0.05)
    assert log.sent == []


@pytest.mark.live
@pytest.mark.skipif(not has_credentials(), reason="no Anthropic credentials configured")
def test_demo_task_live() -> None:
    answer = run(DEMO_TASK)
    assert "53562" in answer.replace(",", ""), answer  # 250000 * 1.115**7 = 535,628.99
