"""A run written step by step to a Markdown file as it happens: each model call with what it was
newly told, what it thought, said and asked for, and what it cost; each tool call with its input
and result. A `tracing.Tracer`, so it sees whatever the Langfuse tracer would.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TextIO, cast

from aigent.llm.messages import Usage
from aigent.tracing import Kind, Observation

_DEPTH: ContextVar[int] = ContextVar("transcript_depth", default=0)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _fenced(text: str) -> str:
    return f"```text\n{text.strip()}\n```"


def _blocks(content: object) -> list[Mapping[str, object]]:
    """A message's content as blocks; a plain string is one text block."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, Sequence):
        return [b for b in cast(Sequence[object], content) if isinstance(b, Mapping)]
    return []


def rendered(blocks: Sequence[Mapping[str, object]]) -> list[str]:
    """Blocks as Markdown paragraphs: thinking, text, tool calls and tool results."""
    out: list[str] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "thinking" and str(block.get("thinking", "")).strip():
            out.append(f"**Thought:**\n\n{_fenced(str(block['thinking']))}")
        elif kind == "text" and str(block.get("text", "")).strip():
            out.append(f"**Said:**\n\n{str(block['text']).strip()}")
        elif kind == "tool_use":
            leaked = " *(salvaged from leaked text)*" if block.get("salvaged") else ""
            out.append(f"**Asked for** `{block.get('name')}` `{_json(block.get('input'))}`{leaked}")
    return out


class _Seen:
    """One open observation, writing what it finishes with."""

    def __init__(self, tracer: FileTracer, kind: Kind) -> None:
        self._tracer = tracer
        self._kind = kind

    def finish(
        self,
        output: object = None,
        *,
        error: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if error is not None:
            self._tracer.write(f"**Error:**\n\n{_fenced(error)}")
        elif self._kind == "generation" and isinstance(output, Mapping):
            reply = cast(Mapping[str, object], output)
            parts = rendered(_blocks(reply.get("content")))
            self._tracer.write(*parts, f"*stopped on {reply.get('stop_reason')}*")
        elif output is not None:
            text = output if isinstance(output, str) else _json(output)
            self._tracer.write(f"**Result:**\n\n{_fenced(text)}")

    def bill(self, model: str, usage: Usage, usd: float) -> None:
        self._tracer.write(
            f"*{model}: {usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens} "
            f"in ({usage.cache_read_tokens} cached), {usage.output_tokens} out, ${usd:.5f}*"
        )

    def score(self, name: str, passed: bool, comment: str = "") -> None:
        self._tracer.write(f"**Score** {name}: {'pass' if passed else 'fail'} {comment}".strip())


class FileTracer:
    """Every observation appended to `path` as it opens and finishes, nested by heading level,
    each write flushed so the file can be read while the run goes on."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._systems: set[str] = set()

    def write(self, *paragraphs: str) -> None:
        """Paragraphs appended to the file, and flushed."""
        with self._lock:
            for paragraph in paragraphs:
                self._file.write(f"{paragraph}\n\n")
            self._file.flush()

    @contextmanager
    def _opened(
        self, name: str, kind: Kind, inputs: object, metadata: Mapping[str, object] | None
    ) -> Iterator[Observation]:
        depth = _DEPTH.get()
        level = "#" * min(depth + 2, 6)
        detail = f" · {_json(dict(metadata))}" if metadata else ""
        self.write(f"{level} {kind}: {name}{detail}")
        if kind == "generation":
            self._told(inputs)
        elif inputs is not None:
            self.write(f"**Input** `{_json(inputs)}`")
        token = _DEPTH.set(depth + 1)
        try:
            yield _Seen(self, kind)
        except Exception as exc:
            self.write(f"**Raised:**\n\n{_fenced(f'{type(exc).__name__}: {exc}')}")
            raise
        finally:
            _DEPTH.reset(token)

    def _told(self, inputs: object) -> None:
        """A system prompt the first time it is seen, then the newest message if it says
        something; tool results already appear under their tool calls."""
        messages = [
            cast(Mapping[str, object], m)
            for m in (cast(Sequence[object], inputs) if isinstance(inputs, Sequence) else [])
            if isinstance(m, Mapping)
        ]
        for message in messages:
            if message.get("role") == "system":
                system = "\n".join(
                    str(b.get("text", "")) for b in _blocks(message.get("content"))
                ).strip()
                if system and system not in self._systems:
                    self._systems.add(system)
                    self.write(f"**System prompt:**\n\n{_fenced(system)}")
        said = [m for m in messages if m.get("role") != "system"]
        if said:
            newest = _blocks(said[-1].get("content"))
            told = "\n\n".join(str(b.get("text", "")).strip() for b in newest).strip()
            if told:
                self.write(f"**Told:**\n\n{_fenced(told)}")

    def observe(
        self,
        name: str,
        kind: Kind = "span",
        *,
        inputs: object = None,
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] = (),
        session: str | None = None,
    ) -> AbstractContextManager[Observation]:
        return self._opened(name, kind, inputs, metadata)

    def flush(self) -> None:
        with self._lock:
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()
