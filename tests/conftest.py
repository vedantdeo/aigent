"""Fixtures shared across test modules.

The scripted judge and the fake embedder live here because more than one module needs each, and two
copies of a fake drift.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from typing import cast

import anthropic
import numpy as np
import pytest
from anthropic.types import MessageTokensCount, Usage

from entropic.evals.judge import LlmJudge, Verdict
from entropic.week02.embed import Vectors

JUDGE_USAGE = Usage(input_tokens=200, output_tokens=30)
RUBRIC = "The answer must name the company and the direction of the move."


class _Parsed:
    def __init__(self, verdict: Verdict | None) -> None:
        self.parsed_output = verdict
        self.usage = JUDGE_USAGE
        self.stop_reason = "end_turn"


class ScriptedMessages:
    """Stands in for `client.messages`: one scripted verdict, and a log of what was asked."""

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


FAKE_DIMENSIONS = 32


class BagOfWordsEmbedder:
    """Deterministic word-overlap vectors, so a test can say "this query should find that chunk".

    Hashed with CRC32 rather than `hash`, which Python randomises per process. `normalise=False`
    breaks the contract on purpose.
    """

    def __init__(self, *, normalise: bool = True) -> None:
        self.name = "bag-of-words-fake"
        self.dimensions = FAKE_DIMENSIONS
        self._normalise = normalise

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> Vectors:
        return self._vector(text)

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in text.casefold().split():
            vector[zlib.crc32(word.encode()) % self.dimensions] += 1.0
        if not self._normalise:
            return vector + 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return cast(Vectors, vector / norm)


@pytest.fixture
def embedder() -> BagOfWordsEmbedder:
    return BagOfWordsEmbedder()
