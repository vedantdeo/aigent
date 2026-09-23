"""The OpenAI wire: anything serving `/v1/chat/completions` — `mlx_lm.server`, vLLM, OpenAI.

Thinner than the Anthropic one, and honest about it. There is no prompt caching, no server-side
search, no thinking budget and no constrained decoding here, so `llm` refuses a request asking for
one rather than quietly sending something else. A record is *prompted* for and parsed from the
text, which is why `Parsed.parsed` comes back None more often than it does on a wire that
constrains the tokens.

Counting is exact rather than estimated: the served model's tokenizer templates the same messages
the server will, and the ids are counted. Loading it costs a second, once, and only when a call is
actually counted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import openai
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from entropic.adapters.client import Client, model_of
from entropic.errors import Unsupported
from entropic.messages import Block, Parsed, Reply, Usage

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from entropic.adapters import Streamed, ToolSession
    from entropic.llm import Dispatch, Request

# What this wire can do. Everything absent from it — caching, thinking, effort, tools, server-side
# search, streaming — `llm` refuses before sending. `schema` is here because a caller may ask for
# a record, not because the wire constrains one.
SUPPORTS = frozenset({"schema"})

# Local servers want no credential; a real OpenAI endpoint reads OPENAI_API_KEY from the env.
LOCAL_API_KEY = "not-needed"

# The first {...} in a reply, which is where a prompted record lives when the model wrapped it in
# prose or a code fence.
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def usage_of(usage: CompletionUsage | None) -> Usage:
    """An OpenAI usage as ours. A server that reports none is billed as nothing, not guessed at."""
    if usage is None:
        return Usage(input_tokens=0, output_tokens=0)
    details = usage.prompt_tokens_details
    cached = details.cached_tokens or 0 if details else 0
    return Usage(
        input_tokens=usage.prompt_tokens - cached,
        output_tokens=usage.completion_tokens,
        cache_read_tokens=cached,
    )


def reply_of(completion: ChatCompletion) -> Reply:
    """One completion as ours. One choice is asked for, so one is read."""
    choice = completion.choices[0]
    text = choice.message.content or ""
    return Reply(
        text=text,
        usage=usage_of(completion.usage),
        stop_reason=choice.finish_reason,
        model=completion.model,
        blocks=({"type": "text", "text": text},),
        raw=completion,
    )


def schema_instruction(schema: type[BaseModel]) -> str:
    """What replaces constrained decoding: the schema, and an instruction to send only JSON."""
    return (
        "Reply with a single JSON object matching this schema, and nothing else — no prose, no "
        f"code fence, no explanation.\n\n{json.dumps(schema.model_json_schema())}"
    )


def record_in[Record: BaseModel](text: str, schema: type[Record]) -> Record | None:
    """The record a reply holds, or None if it holds none this schema accepts."""
    found = JSON_OBJECT.search(text)
    if found is None:
        return None
    try:
        return schema.model_validate_json(found.group())
    except ValidationError:
        return None


class OpenAI:
    """The OpenAI wire, holding one SDK client pointed wherever its client's settings say."""

    name = "openai"
    supports = SUPPORTS

    def __init__(self, settings: Client, sdk: object | None = None) -> None:
        self.settings = settings
        self._client = cast("openai.OpenAI | None", sdk)
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def client(self) -> openai.OpenAI:
        """Built on first use. A local server takes any key, so one is supplied rather than
        demanded of the environment."""
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.settings.base_url, api_key=LOCAL_API_KEY, max_retries=0
            )
        return self._client

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The served model's own tokenizer, so a count is the number the server will see."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # heavy, and only a count needs it

            self._tokenizer = cast(
                "PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(self.settings.model)
            )
        return self._tokenizer

    def messages(
        self, request: Request, schema: type[BaseModel] | None = None
    ) -> list[ChatCompletionMessageParam]:
        """The request as this wire carries it: one system message, then the turns, with a
        prompted schema appended to the system prompt when a record is wanted."""
        system = _text_of(request.system)
        if schema is not None:
            system = f"{system}\n\n{schema_instruction(schema)}".strip()
        sent: list[ChatCompletionMessageParam] = []
        if system:
            sent.append({"role": "system", "content": system})
        for message in request.messages:
            sent.append(
                cast(
                    ChatCompletionMessageParam,
                    {"role": message["role"], "content": _text_of(message["content"])},
                )
            )
        return sent

    def count(self, request: Request, schema: type[BaseModel] | None = None) -> int:
        """The input tokens this request would send, templated as the server will template them."""
        # return_dict=False or this hands back a BatchEncoding, whose length is its number of
        # fields — a count of 2 for any prompt, which every guard would wave through.
        ids = self.tokenizer.apply_chat_template(
            cast("list[dict[str, str]]", self.messages(request, schema)),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        return len(cast(Sequence[int], ids))

    def send(self, request: Request) -> Reply:
        """One request, one reply."""
        return reply_of(
            self.client.chat.completions.create(
                model=model_of(request, self.settings),
                messages=self.messages(request),
                max_completion_tokens=request.max_tokens,
            )
        )

    def parse[Record: BaseModel](
        self, request: Request, schema: type[Record], cache_body: dict[str, object] | None = None
    ) -> Parsed[Record]:
        """One request for a record, prompted rather than constrained, and read from the text.

        `cache_body` is ignored: this wire has no prompt caching, and `llm` has already refused any
        request that asked for it.
        """
        completion = self.client.chat.completions.create(
            model=model_of(request, self.settings),
            messages=self.messages(request, schema),
            max_completion_tokens=request.max_tokens,
        )
        reply = reply_of(completion)
        return Parsed[Record](
            text=reply.text,
            usage=reply.usage,
            stop_reason=reply.stop_reason,
            model=reply.model,
            blocks=reply.blocks,
            raw=completion,
            parsed=record_in(reply.text, schema),
        )

    @contextmanager
    def streamed(self, request: Request) -> Iterator[Streamed]:
        """Refused for now: a streamed call must still bill exactly, and that wants the usage the
        final chunk carries only when `stream_options` asks for it. Until that is measured rather
        than assumed, this wire does not stream."""
        raise Unsupported(f"{request.step}: the {self.name} wire does not stream yet")
        yield  # pragma: no cover - unreachable, and what makes this a generator

    def tools(self, request: Request, dispatch: Dispatch, max_turns: int) -> ToolSession:
        """Refused: this wire has tool calling, but no runner here drives it yet."""
        raise Unsupported(f"{request.step}: the {self.name} wire runs no tools yet")


def _text_of(content: str | Sequence[Block] | None) -> str:
    """Blocks flattened to the text this wire carries. A block with no text contributes none."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(str(block.get("text", "")) for block in content if block.get("text"))
