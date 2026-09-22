"""Fixtures shared across test modules.

The scripted judge, the fake client, the fake search and the fake embedder live here because more
than one module needs each, and two copies of a fake drift.
"""

from __future__ import annotations

import threading
import zlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import anthropic
import numpy as np
import pytest
from anthropic import Omit
from anthropic.lib.tools import BetaFunctionTool, BetaToolRunner
from anthropic.types import (
    Message,
    MessageTokensCount,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from anthropic.types.beta import BetaMessage
from anthropic.types.beta.message_create_params import ParseMessageCreateParamsBase
from pydantic import BaseModel

from entropic.config import MODEL
from entropic.evals.judge import LlmJudge, Verdict
from entropic.llm import Llm
from entropic.pricing import Budget
from entropic.retrieval.chunk import Chunk
from entropic.retrieval.embed import Vectors

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
        reference: str | None = None,
        fails_with: Exception | None = None,
    ) -> tuple[LlmJudge, ScriptedMessages]:
        fake = _ScriptedClient(verdict, fails_with)
        judge = LlmJudge(rubric=rubric, reference=reference, client=cast(anthropic.Anthropic, fake))
        return judge, fake.messages

    return build


@dataclass(frozen=True)
class Sent:
    """One call as the fake client received it."""

    kind: str
    model: str
    max_tokens: int
    system: object
    messages: list[dict[str, object]]
    schema: type | None
    thinking: object
    tools: object
    cache_control: object

    @property
    def prompt(self) -> str:
        """The last message's text, when it is plain text."""
        content = self.messages[-1]["content"]
        return content if isinstance(content, str) else ""


Reply = Callable[[Sent], str | BaseModel | Exception | None]
FAKE_USAGE = Usage(input_tokens=100, output_tokens=50)


class _ParsedReply:
    def __init__(self, record: BaseModel | None, stop_reason: str) -> None:
        self.parsed_output = record
        self.usage = FAKE_USAGE
        self.stop_reason = stop_reason


class _FakeStream:
    def __init__(self, message: Message) -> None:
        self._message = message

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    @property
    def text_stream(self) -> Iterator[str]:
        for block in self._message.content:
            if block.type == "text":
                yield from block.text.split(" ")

    def get_final_message(self) -> Message:
        return self._message


class FakeMessages:
    """Stands in for `client.messages`, replying through `reply` and logging every call.

    Replies are decided by what was sent rather than by arrival order, so concurrent calls can land
    in any order and still get the answer meant for them. A reply may be text, a whole `Message`
    (to script tool use), or for a parse call a record, None, or an exception to raise.
    `input_tokens` and `count_error` decide what the free count says.
    """

    def __init__(self, reply: Reply, stop_reason: str) -> None:
        self._reply = reply
        self._stop_reason = stop_reason
        self._lock = threading.Lock()
        self.sent: list[Sent] = []
        self.counted: list[type | None] = []
        self.input_tokens = FAKE_USAGE.input_tokens
        self.count_error: Exception | None = None

    def count_tokens(self, **kwargs: object) -> MessageTokensCount:
        with self._lock:
            self.counted.append(_schema(kwargs))
        if self.count_error is not None:
            raise self.count_error
        return MessageTokensCount(input_tokens=self.input_tokens)

    def create(self, **kwargs: object) -> Message:
        return self._message(self._reply(self._log("create", kwargs)), kwargs)

    def stream(self, **kwargs: object) -> _FakeStream:
        return _FakeStream(self._message(self._reply(self._log("stream", kwargs)), kwargs))

    def turn(self, **kwargs: object) -> BetaMessage:
        """One tool-runner turn: the runner calls `beta.messages.parse`, which lands here."""
        message = self._message(self._reply(self._log("turn", kwargs)), kwargs)
        return BetaMessage.model_validate(message.model_dump())

    def parse(self, **kwargs: object) -> _ParsedReply:
        record = self._reply(self._log("parse", kwargs))
        if isinstance(record, Exception):
            raise record
        assert not isinstance(record, str), f"a parse call wants a record, not {record!r}"
        return _ParsedReply(record, self._stop_reason)

    def _message(
        self, reply: str | BaseModel | Exception | None, kwargs: dict[str, object]
    ) -> Message:
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, Message):
            return reply
        assert isinstance(reply, str), f"a create call is answered with text, not {reply!r}"
        return Message(
            id="msg_fake",
            type="message",
            role="assistant",
            model=str(kwargs["model"]),
            content=[TextBlock(type="text", text=reply)],
            stop_reason=cast(StopReason, self._stop_reason),
            stop_sequence=None,
            usage=FAKE_USAGE,
        )

    def _log(self, kind: str, kwargs: dict[str, object]) -> Sent:
        sent = Sent(
            kind=kind,
            model=str(kwargs["model"]),
            max_tokens=cast(int, kwargs["max_tokens"]),
            system=kwargs.get("system"),
            messages=list(cast(Sequence[dict[str, object]], kwargs["messages"])),
            schema=_schema(kwargs),
            thinking=kwargs.get("thinking"),
            tools=kwargs.get("tools"),
            cache_control=_cache_control(kwargs),
        )
        with self._lock:
            self.sent.append(sent)
        return sent


def _cache_control(kwargs: dict[str, object]) -> object:
    """Top-level caching as the wire sees it, whether passed by name or, for parse, in the body."""
    body = cast(dict[str, object], kwargs.get("extra_body") or {})
    value = kwargs.get("cache_control", body.get("cache_control"))
    return None if isinstance(value, Omit) else value


def _schema(kwargs: dict[str, object]) -> type | None:
    output_format = kwargs.get("output_format")
    return output_format if isinstance(output_format, type) else None


class _FakeBetaMessages:
    """`client.beta.messages`: `tool_runner` builds the SDK's own runner over this fake, so a test
    exercises the real loop, and each turn it sends comes back through `FakeMessages.turn`."""

    def __init__(self, client: FakeAnthropic) -> None:
        self._client = client

    def tool_runner(
        self,
        *,
        tools: Sequence[BetaFunctionTool[Callable[..., str]]],
        max_iterations: int | None = None,
        **params: object,
    ) -> BetaToolRunner[None]:
        sent = cast(
            ParseMessageCreateParamsBase[None], {**params, "tools": [t.to_dict() for t in tools]}
        )
        client = cast(anthropic.Anthropic, self._client)
        return BetaToolRunner(
            params=sent, options={}, tools=tools, client=client, max_iterations=max_iterations
        )

    def parse(self, **kwargs: object) -> BetaMessage:
        return self._client.messages.turn(**kwargs)


class _FakeBeta:
    def __init__(self, client: FakeAnthropic) -> None:
        self.messages = _FakeBetaMessages(client)


class FakeAnthropic:
    """A client for code that sends several different calls: see `FakeMessages`."""

    def __init__(self, reply: Reply, *, stop_reason: str = "end_turn") -> None:
        self.messages = FakeMessages(reply, stop_reason)
        self.beta = _FakeBeta(self)


def tool_turn(*calls: tuple[str, str, dict[str, object]]) -> Message:
    """An assistant turn asking for tools: `(id, name, input)` per call."""
    blocks = [ToolUseBlock(type="tool_use", id=i, name=name, input=args) for i, name, args in calls]
    return Message(
        id="msg_tool",
        type="message",
        role="assistant",
        model=MODEL,
        content=list(blocks),
        stop_reason="tool_use",
        stop_sequence=None,
        usage=FAKE_USAGE,
    )


def turns(*replies: Message | str) -> Reply:
    """The first request gets the first reply, the next the second: each turn adds two messages."""

    def reply(sent: Sent) -> Message | str:
        return replies[min((len(sent.messages) - 1) // 2, len(replies) - 1)]

    return reply


def tool_results(sent: Sent) -> list[dict[str, object]]:
    """The tool results a turn carried back: its last message's blocks."""
    return cast(list[dict[str, object]], sent.messages[-1]["content"])


MakeLlm = Callable[..., tuple[Llm, FakeMessages]]


@pytest.fixture
def make_llm() -> MakeLlm:
    """An `Llm` whose replies `reply` decides, and the log of what it sent."""

    def build(
        reply: Reply,
        *,
        stop_reason: str = "end_turn",
        limit_usd: float = 1.0,
        rehearse: bool = False,
    ) -> tuple[Llm, FakeMessages]:
        fake = FakeAnthropic(reply, stop_reason=stop_reason)
        budget = Budget(limit_usd=limit_usd)
        return Llm(cast(anthropic.Anthropic, fake), budget=budget, rehearse=rehearse), fake.messages

    return build


def _passage(doc_id: str, text: str) -> Chunk:
    return Chunk(id=f"{doc_id}#0001", text=text, doc_id=doc_id, ordinal=1, start=0, page=12)


PASSAGES: dict[str, list[Chunk]] = {
    "ITC-FY25": [
        _passage("ITC-FY25", "Cigarettes net segment revenue grew 6.5 per cent during the year.")
    ],
    "RELIANCE-FY25": [
        _passage("RELIANCE-FY25", "The Board recommended a dividend of Rs 5.50 per equity share.")
    ],
    "TATAMOTORS-FY25": [
        _passage("TATAMOTORS-FY25", "Jaguar Land Rover delivered record free cash flow in FY25.")
    ],
}


class FakeSearch:
    """Every report's passages when unscoped, one report's when scoped; every query logged."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str | None]] = []

    def __call__(self, query: str, /, *, doc_id: str | None = None) -> list[Chunk]:
        self.asked.append((query, doc_id))
        if doc_id is None:
            return [chunk for chunks in PASSAGES.values() for chunk in chunks]
        return list(PASSAGES[doc_id])


@pytest.fixture
def search() -> FakeSearch:
    return FakeSearch()


class KeywordReranker:
    """A deterministic stand-in for a cross-encoder: score by query words the passage contains.

    Crude, and crucially **a different ordering from `BagOfWordsEmbedder`'s** — a fake reranker
    that agreed with the fake retriever could not show that reranking changed anything.
    """

    def __init__(self, *, name: str = "keyword-rerank-fake") -> None:
        self.name = name
        self.seen: list[tuple[str, int]] = []

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        self.seen.append((query, len(passages)))
        terms = set(query.casefold().split())
        return [float(sum(word in terms for word in p.casefold().split())) for p in passages]


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
