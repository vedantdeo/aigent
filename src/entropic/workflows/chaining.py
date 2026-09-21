"""Prompt chaining: a fixed sequence of calls, with a check in code between two of them.

uv run python -m entropic.workflows.chaining ["question"] [--yes]

Extract facts with verbatim quotes, keep only those whose quote really is in the passage it cites,
then write a note from the survivors. The gate is code: a quote is in its passage or it is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from entropic.config import MAX_TOKENS_FACTS, MAX_TOKENS_NOTE
from entropic.config import THINKING_WORKFLOW_PARAM as THINKING
from entropic.llm import Llm, Request
from entropic.retrieval.chunk import Chunk, context_block, squeeze
from entropic.workflows.demo import run_demo
from entropic.workflows.reports import Search

QUESTIONS = ("How did ITC's cigarettes business perform in FY25?",)

# Below this a quote proves nothing: "the company" is in every passage.
_MIN_QUOTE_WORDS = 3

EXTRACT = (
    "You extract facts from annual-report passages. Copy each quote character for character "
    "from the passage it cites; a quote that is not in its passage is discarded."
)
WRITE = (
    "You write short investor notes from a list of verified facts. Use only those facts, cite "
    "each by its passage id in square brackets, and stay under 120 words."
)


class Fact(BaseModel):
    claim: str = Field(description="One factual statement, in your own words.")
    quote: str = Field(description="The words in the passage that support it, copied exactly.")
    passage: str = Field(description="The id of the passage the quote is from.")


class Facts(BaseModel):
    facts: list[Fact] = Field(description="At most six, most relevant first; empty if none.")


@dataclass(frozen=True)
class Chained:
    note: str | None
    kept: list[Fact]
    dropped: list[Fact]


def gate(facts: Sequence[Fact], passages: Sequence[Chunk]) -> tuple[list[Fact], list[Fact]]:
    """Split facts into those whose quote is in the passage they cite, and the rest."""
    texts = {chunk.id: squeeze(chunk.text) for chunk in passages}
    kept: list[Fact] = []
    dropped: list[Fact] = []
    for fact in facts:
        quote = squeeze(fact.quote)
        held = len(quote.split()) >= _MIN_QUOTE_WORDS and quote in texts.get(fact.passage, "")
        (kept if held else dropped).append(fact)
    return kept, dropped


def run(llm: Llm, search: Search, question: str) -> Chained:
    passages = search(question)
    prompt = f"{context_block(passages)}\n\nquestion: {question}"
    extract = Request.ask("extract", EXTRACT, prompt, MAX_TOKENS_FACTS, thinking=THINKING)
    kept, dropped = gate(llm.record(extract, Facts).facts, passages)
    if not kept:
        return Chained(None, kept, dropped)
    listed = "\n".join(f"- {fact.claim} [{fact.passage}]" for fact in kept)
    prompt = f"question: {question}\n\nfacts:\n{listed}"
    write = Request.ask("write", WRITE, prompt, MAX_TOKENS_NOTE, thinking=THINKING)
    return Chained(llm.text(write), kept, dropped)


def show(result: Chained) -> str:
    lines = [f"gate: kept {len(result.kept)} facts, dropped {len(result.dropped)}"]
    lines += [f"  dropped [{fact.passage}]: {fact.quote!r}" for fact in result.dropped]
    lines.append(f"\n{result.note or '(nothing survived the gate, so no note was written)'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    run_demo(argv, prog="entropic.workflows.chaining", questions=QUESTIONS, run=run, show=show)


if __name__ == "__main__":
    main()
