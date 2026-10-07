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
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast, get_args, get_origin

import openai
from openai import Omit, omit
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel, ValidationError

from aigent.errors import Unsupported
from aigent.llm.adapters.client import Client, model_of
from aigent.llm.adapters.retry import http_client
from aigent.llm.messages import BatchStatus, Block, Failed, Parsed, Reply, Usage

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedTokenizerBase

    from aigent.llm.adapters import Streamed, ToolSession
    from aigent.llm.core import Dispatch, Request

# What this wire can do. Everything absent from it — caching, thinking, effort, tools, server-side
# search, streaming — `llm` refuses before sending. `schema` is here because a caller may ask for
# a record, not because the wire constrains one.
SUPPORTS = frozenset({"schema"})

# Local servers want no credential; a hosted endpoint names the variable holding its key.
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


# The OpenAI finish reasons `llm` acts on, under the names it checks; the rest pass through.
STOP_REASONS = {"content_filter": "refusal", "length": "max_tokens", "tool_calls": "tool_use"}


def reply_of(completion: ChatCompletion) -> Reply:
    """One completion as ours. One choice is asked for, so one is read."""
    choice = completion.choices[0]
    text = choice.message.content or ""
    return Reply(
        text=text,
        usage=usage_of(completion.usage),
        stop_reason=STOP_REASONS.get(choice.finish_reason, choice.finish_reason),
        model=completion.model,
        blocks=({"type": "text", "text": text},),
        raw=completion,
    )


def _plainly(annotation: object) -> str:
    """A field's type as a person would say it, not as typing prints it."""
    if get_origin(annotation) is list:
        return f"array of {_plainly(get_args(annotation)[0])}s"
    named = {bool: "boolean", str: "string", int: "number", float: "number"}
    return named.get(cast(type, annotation), getattr(annotation, "__name__", str(annotation)))


def schema_instruction(schema: type[BaseModel]) -> str:
    """What replaces constrained decoding: the fields wanted, and an order to send only JSON.

    The fields are described rather than handed over as a JSON Schema document, because a model
    given a schema answers with the schema. Measured on Qwen3-8B over twelve questions: a bare
    schema parsed 0 of 12, two thirds of them echoing it back with the values inside `properties`;
    describing the keys parsed 12 of 12, and was the faster of the two for having less to read
    and less to write.
    """
    fields = "\n".join(
        f"- {name} ({_plainly(field.annotation)}): {field.description or ''}".rstrip()
        for name, field in schema.model_fields.items()
    )
    return (
        "Reply with a single JSON object and nothing else — no prose, no code fence, no "
        f"explanation. It has exactly these keys:\n{fields}"
    )


def record_in[Record: BaseModel](text: str, schema: type[Record]) -> Record | None:
    """The record a reply holds, or None if it holds none this schema accepts.

    A model handed a JSON Schema sometimes answers with the schema itself, values filled into its
    `properties`, which is a right answer in a wrong envelope. That shape is unwrapped rather than
    thrown away; anything else that does not validate is None.
    """
    found = JSON_OBJECT.search(text)
    if found is None:
        return None
    try:
        return schema.model_validate_json(found.group())
    except ValidationError:
        pass
    try:
        echoed = json.loads(found.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(echoed, dict):
        return None
    inside = echoed.get("properties")
    if not isinstance(inside, dict):
        return None
    # Deliberately not also requiring `"type": "object"`: a model that echoes the schema drops
    # parts of it at will, and that key's absence cost 27 of 54 rows on a run that had answered
    # them correctly. Validation below is what makes the unwrap safe, not the shape of the wrapper.
    try:
        return schema.model_validate(inside)
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
        """Built on first use, with the key its client names; a local server takes any key."""
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.settings.base_url,
                api_key=self._api_key(),
                max_retries=0,
                timeout=self.settings.timeout,
                http_client=http_client(
                    self.settings.timeout,
                    self.settings.keepalive_seconds,
                    self.settings.retries,
                    self.settings.retry_base_seconds,
                    self.settings.retry_max_seconds,
                ),
            )
        return self._client

    def _api_key(self) -> str:
        """The key from the environment variable this client names. Raises if it is unset."""
        variable = self.settings.api_key_env
        if variable is None:
            return LOCAL_API_KEY
        key = os.environ.get(variable)
        if not key:
            raise RuntimeError(f"{self.settings.name} needs {variable} set; add it to .env")
        return key

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The served model's own tokenizer, so a count is the number the server will see."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # heavy, and only a count needs it

            repo = self.settings.tokenizer or self.settings.model
            self._tokenizer = cast(
                "PreTrainedTokenizerBase",
                AutoTokenizer.from_pretrained(repo, trust_remote_code=False),
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

    @property
    def _extra_body(self) -> dict[str, object] | None:
        """What the server needs beyond the OpenAI fields: how to template this model, and the
        adapter to apply to it."""
        body: dict[str, object] = dict(self.settings.extra_body)
        if self.settings.template_kwargs:
            body["chat_template_kwargs"] = dict(self.settings.template_kwargs)
        if self.settings.adapter is not None:
            body["adapters"] = self.settings.adapter
        return body or None

    def _response_format(self, schema: type[BaseModel]) -> ResponseFormatJSONSchema | Omit:
        """The schema for the server to enforce, or nothing where it only follows the prompt."""
        if not self.settings.constrains_schema:
            return omit
        return {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }

    def count(self, request: Request, schema: type[BaseModel] | None = None) -> int:
        """The input tokens this request would send, templated as the server will template them."""
        # return_dict=False or this hands back a BatchEncoding, whose length is its number of
        # fields — a count of 2 for any prompt, which every guard would wave through.
        # Cast because the stub names `tools` and `documents` as parameters, which a client's
        # own template arguments would be matched against.
        template = cast("Callable[..., Sequence[int]]", self.tokenizer.apply_chat_template)
        ids = template(
            self.messages(request, schema),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **self.settings.template_kwargs,
        )
        return len(ids)

    def send(self, request: Request) -> Reply:
        """One request, one reply."""
        return reply_of(
            self.client.chat.completions.create(
                model=model_of(request, self.settings),
                messages=self.messages(request),
                max_tokens=request.max_tokens,
                extra_body=self._extra_body,
            )
        )

    def parse[Record: BaseModel](
        self, request: Request, schema: type[Record], cache_body: dict[str, object] | None = None
    ) -> Parsed[Record]:
        """One request for a record, prompted, and constrained too where the client's server
        enforces a schema; read from the text either way.

        `cache_body` is ignored: this wire has no prompt caching, and `llm` has already refused any
        request that asked for it.
        """
        completion = self.client.chat.completions.create(
            model=model_of(request, self.settings),
            messages=self.messages(request, schema),
            max_tokens=request.max_tokens,
            response_format=self._response_format(schema),
            extra_body=self._extra_body,
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

    def submit(self, batch: Sequence[tuple[str, Request]], schema: type[BaseModel]) -> str:
        """Refused: no server this wire talks to has a batch endpoint wired up here."""
        raise Unsupported(f"the {self.name} wire sends no batches")

    def poll(self, batch_id: str) -> BatchStatus:
        raise Unsupported(f"the {self.name} wire sends no batches")

    def collect[Record: BaseModel](
        self, batch_id: str, schema: type[Record]
    ) -> dict[str, Parsed[Record] | Failed]:
        raise Unsupported(f"the {self.name} wire sends no batches")


def _text_of(content: str | Sequence[Block] | None) -> str:
    """Blocks flattened to the text this wire carries. A block with no text contributes none."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(str(block.get("text", "")) for block in content if block.get("text"))
