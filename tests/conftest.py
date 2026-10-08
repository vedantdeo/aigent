"""Fixtures shared across test modules.

The scripted judge, the fake client, the fake search, the fake embedder and the throwaway sandbox
live here because more
than one module needs each, and two copies of a fake drift.
"""

from __future__ import annotations

import contextvars
import threading
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
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
from anthropic.types.beta import BetaMessage, BetaToolUnionParam
from anthropic.types.beta.message_create_params import ParseMessageCreateParamsBase
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse
from pydantic import BaseModel

from aigent import tracing
from aigent.config import CLIENT, JUDGE_MODEL, MAX_USD_PER_TURN, MODEL
from aigent.evals.judge import LlmJudge, Verdict
from aigent.llm.adapters import CLIENTS
from aigent.llm.adapters.cfg_anthropic import CLIENT as ANTHROPIC
from aigent.llm.adapters.client import Client
from aigent.llm.core import Llm
from aigent.llm.messages import Usage as NeutralUsage
from aigent.llm.pricing import PRICES, Budget
from aigent.retrieval.chunk import Chunk
from aigent.retrieval.embed import Vectors

JUDGE_USAGE = Usage(input_tokens=200, output_tokens=30)


RUBRIC = "The answer must name the company and the direction of the move."


class ParsedReply:
    """What the SDK's `parse` returns, as far as the Anthropic adapter reads it.

    One copy, shared: three modules grew their own, and each had to be found again the day the
    adapter started reading a field none of them had.
    """

    def __init__(
        self,
        record: BaseModel | None,
        *,
        stop_reason: str = "end_turn",
        model: str = MODEL,
        usage: Usage = JUDGE_USAGE,
    ) -> None:
        self.parsed_output = record
        self.usage = usage
        self.stop_reason = stop_reason
        self.model = model
        self.content: list[TextBlock] = []


class ScriptedMessages:
    """Stands in for `client.messages`: one scripted verdict, and a log of what was asked."""

    def __init__(self, verdict: Verdict | None, fails_with: Exception | None) -> None:
        self._verdict = verdict
        self._fails_with = fails_with
        self.usage = JUDGE_USAGE
        self.prompts: list[str] = []
        self.thinking: list[object] = []
        self.caps: list[int] = []
        self.counted = 0

    def count_tokens(self, **_: object) -> MessageTokensCount:
        if self._fails_with is not None:
            raise self._fails_with
        self.counted += 1
        return MessageTokensCount(input_tokens=JUDGE_USAGE.input_tokens)

    def parse(self, **kwargs: object) -> ParsedReply:
        messages = cast("list[dict[str, str]]", kwargs["messages"])
        self.prompts.append(messages[0]["content"])
        self.thinking.append(kwargs.get("thinking"))
        self.caps.append(cast(int, kwargs["max_tokens"]))
        return ParsedReply(self._verdict, model=JUDGE_MODEL)


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
        client: str = CLIENT,
    ) -> tuple[LlmJudge, ScriptedMessages]:
        fake = _ScriptedClient(verdict, fails_with)
        judge = LlmJudge(rubric=rubric, reference=reference, client=client, sdk=fake)
        return judge, fake.messages

    return build


# The Anthropic client under another name, with caps that match no default, so a test can tell
# which client's cap a call went out with. Registered only for a test that asks for `capped`.
CAPPED = replace(
    ANTHROPIC, name="capped", max_tokens={"ANSWER": 300, "HEADLINE": 100, "JUDGE": 200}
)


@pytest.fixture
def capped(monkeypatch: pytest.MonkeyPatch) -> Client:
    monkeypatch.setitem(CLIENTS, CAPPED.name, CAPPED)
    return CAPPED


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


def context_costing(usd: float) -> int:
    """The cached context whose write alone costs `usd`: so a row reads relative to the ceiling."""
    return int(usd / (PRICES[MODEL].cache_write * 1e-6))


# Past the per-turn ceiling with the output cap on top, with $0.05 left for an answer.
GROWN_PAST = context_costing(MAX_USD_PER_TURN - 0.05)


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
        self.batches = FakeBatches(self)

    def count_tokens(self, **kwargs: object) -> MessageTokensCount:
        with self._lock:
            self.counted.append(_schema(kwargs))
        if self.count_error is not None:
            raise self.count_error
        return MessageTokensCount(input_tokens=self.input_tokens)

    def create(self, **kwargs: object) -> Message:
        return self._message(self._reply(self._log("create", kwargs)), kwargs)

    def reply_to(self, sent: Sent) -> str | BaseModel | Exception | None:
        return self._reply(sent)

    def log(self, kind: str, kwargs: dict[str, object]) -> Sent:
        return self._log(kind, kwargs)

    def message(self, text: str, kwargs: dict[str, object]) -> Message:
        return self._message(text, kwargs)

    def stream(self, **kwargs: object) -> _FakeStream:
        return _FakeStream(self._message(self._reply(self._log("stream", kwargs)), kwargs))

    def turn(self, **kwargs: object) -> BetaMessage:
        """One tool-runner turn: the runner calls `beta.messages.parse`, which lands here."""
        message = self._message(self._reply(self._log("turn", kwargs)), kwargs)
        return BetaMessage.model_validate(message.model_dump())

    def parse(self, **kwargs: object) -> ParsedReply:
        record = self._reply(self._log("parse", kwargs))
        if isinstance(record, Exception):
            raise record
        assert not isinstance(record, str), f"a parse call wants a record, not {record!r}"
        return ParsedReply(
            record, stop_reason=self._stop_reason, model=str(kwargs["model"]), usage=FAKE_USAGE
        )

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


class FakeBatches:
    """`client.messages.batches`: each request is answered through its client's `reply`, so a batch
    is scripted exactly as single calls are. A record succeeds, text succeeds unparseable, an
    exception errors and None expires. The first `polls` checks find the batch still processing,
    and results come back in reverse, as the API may return them in any order."""

    def __init__(self, messages: FakeMessages) -> None:
        self._messages = messages
        self.created: list[list[dict[str, object]]] = []
        self.polls = 0
        self.checked = 0
        self.submit_error: Exception | None = None
        self.results_error: Exception | None = None

    def create(self, *, requests: Sequence[Mapping[str, object]]) -> MessageBatch:
        if self.submit_error is not None:
            raise self.submit_error
        self.created.append([dict(request) for request in requests])
        return self._batch("in_progress")

    def retrieve(self, batch_id: str) -> MessageBatch:
        self.checked += 1
        return self._batch("ended" if self.checked > self.polls else "in_progress")

    def results(self, batch_id: str) -> list[MessageBatchIndividualResponse]:
        if self.results_error is not None:
            raise self.results_error
        return [self._result(request) for request in reversed(self.created[-1])]

    def _result(self, request: dict[str, object]) -> MessageBatchIndividualResponse:
        params = cast(dict[str, object], request["params"])
        reply = self._messages.reply_to(self._messages.log("batch", params))
        if reply is None:
            result: dict[str, object] = {"type": "expired"}
        elif isinstance(reply, Exception):
            error = {"type": "invalid_request_error", "message": str(reply)}
            result = {"type": "errored", "error": {"type": "error", "error": error}}
        else:
            text = reply if isinstance(reply, str) else reply.model_dump_json()
            message = self._messages.message(text, params)
            result = {"type": "succeeded", "message": message.model_dump()}
        body = {"custom_id": request["custom_id"], "result": result}
        return MessageBatchIndividualResponse.model_validate(body)

    def _batch(self, status: str) -> MessageBatch:
        submitted = len(self.created[-1]) if self.created else 0
        ended = status == "ended"
        counts = {"processing": 0 if ended else submitted, "succeeded": submitted if ended else 0}
        return MessageBatch.model_validate(
            {
                "id": f"msgbatch_{len(self.created)}",
                "type": "message_batch",
                "processing_status": status,
                "request_counts": {**counts, "errored": 0, "expired": 0, "canceled": 0},
                "created_at": "2026-10-07T00:00:00Z",
                "expires_at": "2026-10-08T00:00:00Z",
            }
        )


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
        tools: Sequence[BetaFunctionTool[Callable[..., str]] | BetaToolUnionParam],
        max_iterations: int | None = None,
        **params: object,
    ) -> BetaToolRunner[None]:
        # As the SDK does: the tools it runs go on the wire first, then the ones the API runs.
        runnable = [tool for tool in tools if isinstance(tool, BetaFunctionTool)]
        raw = [tool for tool in tools if not isinstance(tool, BetaFunctionTool)]
        wire = [*(tool.to_dict() for tool in runnable), *raw]
        sent = cast(ParseMessageCreateParamsBase[None], {**params, "tools": wire})
        client = cast(anthropic.Anthropic, self._client)
        return BetaToolRunner(
            params=sent, options={}, tools=runnable, client=client, max_iterations=max_iterations
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


def turns(*replies: Message | str | Exception) -> Reply:
    """The first request gets the first reply, the next the second: each turn adds two messages."""

    def reply(sent: Sent) -> Message | str | Exception:
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
        return Llm(sdk=fake, budget=budget, rehearse=rehearse), fake.messages

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


@pytest.fixture
def box(tmp_path: Path) -> Path:
    """A throwaway sandbox with one file inside it and one file outside it."""
    sandbox = tmp_path / "box"
    sandbox.mkdir()
    (sandbox / "notes.txt").write_text("hello from the sandbox\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret\n", encoding="utf-8")
    return sandbox


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


@dataclass
class Observed:
    """One node a `Recorder` saw opened: what it was given, how it finished, and what it held."""

    name: str
    kind: tracing.Kind
    inputs: object = None
    metadata: dict[str, object] = field(default_factory=dict[str, object])
    tags: tuple[str, ...] = ()
    session: str | None = None
    output: object = None
    error: str | None = None
    billed: tuple[str, NeutralUsage, float] | None = None
    scores: dict[str, tuple[bool, str]] = field(default_factory=dict[str, tuple[bool, str]])
    children: list[Observed] = field(default_factory=list["Observed"])

    def finish(
        self,
        output: object = None,
        *,
        error: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.output, self.error = output, error
        self.metadata.update(metadata or {})

    def bill(self, model: str, usage: NeutralUsage, usd: float) -> None:
        self.billed = (model, usage, usd)

    def score(self, name: str, passed: bool, comment: str = "") -> None:
        self.scores[name] = (passed, comment)

    def walk(self) -> Iterator[Observed]:
        yield self
        for child in self.children:
            yield from child.walk()


class Recorder:
    """A tracer that keeps the tree in memory, nesting through context as Langfuse's does."""

    def __init__(self) -> None:
        self.roots: list[Observed] = []
        self._open: contextvars.ContextVar[Observed | None] = contextvars.ContextVar(
            "open", default=None
        )
        self._lock = threading.Lock()

    @contextmanager
    def observe(
        self,
        name: str,
        kind: tracing.Kind = "span",
        *,
        inputs: object = None,
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] = (),
        session: str | None = None,
    ) -> Iterator[tracing.Observation]:
        node = Observed(name, kind, inputs, dict(metadata or {}), tuple(tags), session)
        parent = self._open.get()
        with self._lock:
            (self.roots if parent is None else parent.children).append(node)
        token = self._open.set(node)
        try:
            yield node
        except Exception as exc:
            node.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._open.reset(token)

    def flush(self) -> None:
        pass

    def every(self) -> list[Observed]:
        """Every node, depth first, roots in the order they opened."""
        return [node for root in self.roots for node in root.walk()]


@pytest.fixture(autouse=True)
def traces() -> Iterator[Recorder]:
    """Every test traces into memory, so none can reach Langfuse even with keys in `.env`."""
    recorder = Recorder()
    tracing.use(recorder)
    yield recorder
    tracing.use(None)
