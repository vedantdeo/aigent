"""Fixtures shared across test modules.

The scripted judge lives here because two modules need it for different reasons: the grader tests
check what the judge *says*, the runner tests check that the runner bills and survives it. One fake
with one set of token counts keeps the two from drifting apart.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import anthropic
import pytest
from anthropic.types import MessageTokensCount, Usage

from entropic.evals.judge import LlmJudge, Verdict

JUDGE_USAGE = Usage(input_tokens=200, output_tokens=30)
RUBRIC = "The answer must name the company and the direction of the move."


class _Parsed:
    def __init__(self, verdict: Verdict | None) -> None:
        self.parsed_output = verdict
        self.usage = JUDGE_USAGE
        self.stop_reason = "end_turn"


class ScriptedMessages:
    """Stands in for `client.messages`: one scripted verdict, and a log of what was asked.

    `counted`, `prompts` and `usage` are what tests assert against: that a paid call was
    pre-flighted, what the judge was shown, and what its verdict should cost.
    """

    def __init__(self, verdict: Verdict | None, fails_with: Exception | None) -> None:
        self._verdict = verdict
        self._fails_with = fails_with
        self.usage = JUDGE_USAGE
        self.prompts: list[str] = []
        self.counted = 0

    def count_tokens(self, **_: object) -> MessageTokensCount:
        if self._fails_with is not None:
            raise self._fails_with
        self.counted += 1
        return MessageTokensCount(input_tokens=JUDGE_USAGE.input_tokens)

    def parse(self, **kwargs: object) -> _Parsed:
        messages = cast("list[dict[str, str]]", kwargs["messages"])
        self.prompts.append(messages[0]["content"])
        return _Parsed(self._verdict)


class _ScriptedClient:
    def __init__(self, verdict: Verdict | None, fails_with: Exception | None) -> None:
        self.messages = ScriptedMessages(verdict, fails_with)


MakeJudge = Callable[..., tuple[LlmJudge, ScriptedMessages]]


@pytest.fixture
def make_judge() -> MakeJudge:
    """Build an `LlmJudge` whose verdict is decided here, so no test of it costs anything."""

    def build(
        verdict: Verdict | None = None,
        *,
        rubric: str = RUBRIC,
        fails_with: Exception | None = None,
    ) -> tuple[LlmJudge, ScriptedMessages]:
        fake = _ScriptedClient(verdict, fails_with)
        judge = LlmJudge(rubric=rubric, client=cast(anthropic.Anthropic, fake))
        return judge, fake.messages

    return build
