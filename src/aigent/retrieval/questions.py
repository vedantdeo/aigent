"""Generating the retrieval question set from the corpus.

uv run python -m aigent.retrieval.questions          # counts tokens, spends nothing
uv run python -m aigent.retrieval.questions --yes    # writes evals/datasets/retrieval.jsonl

One question per sampled passage, each labelled with the sentence that answers it. Every quote is
checked back against the corpus before the row is written: a quote the model paraphrased resolves to
no chunk under any strategy, and a quote two reports share resolves to chunks in both. Neither is a
retrieval failure, and both read as one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from aigent.config import MAX_TOKENS_QUESTION as MAX_TOKENS
from aigent.config import (
    MAX_USD_PER_EVAL,
    MODEL,
    PASSAGE_MIN_ALPHA,
    PASSAGE_MIN_CHARS,
    THINKING_EVAL_PARAM,
)
from aigent.llm import Llm, Request
from aigent.messages import Msg
from aigent.pricing import Budget, estimate_eval_usd, usage_cost
from aigent.retrieval.chunk import Chunk, Document, Inventory, by_sentence, squeeze
from aigent.retrieval.corpus import MANIFEST, load_corpus

REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "evals" / "datasets" / "retrieval.jsonl"

SYSTEM = """You write evaluation questions for a document retrieval system, from the annual \
reports of Indian listed companies.

You are given one passage from one company's report. Write one question that the passage answers, \
and copy the exact sentence from the passage that answers it.

- The question must name the company. The corpus holds several annual reports, and an unqualified \
question has several correct answers.
- The question must be answerable from this passage alone, by someone who has not seen it.
- Ask about the fact, never about the text. Not "what does the passage say about capex".
- The quote must be copied character for character from the passage: one sentence, or two if one \
will not do. Do not paraphrase, summarise, correct spelling or spacing, or reflow it. PDF \
extraction leaves odd spacing in this text; copy it as it is rather than tidying it.
- Prefer a specific figure, date, name or commitment over a general statement.
- Set `usable` to false when the passage is a table, a run of headings, or holds no fact worth \
asking about. Leave the other fields empty when you do."""


class Question(BaseModel):
    """One generated question, before it has been checked against the document."""

    usable: bool = Field(description="False when the passage holds no fact worth asking about.")
    question: str = Field(description="Names the company, answerable from the passage alone.")
    quote: str = Field(description="Copied character for character from the passage.")
    topic: str = Field(description="One or two words for slicing the table: dividend, capex, ...")


@dataclass(frozen=True)
class Generated:
    """One model response and what it cost, whether or not the question survived checking."""

    chunk: Chunk
    question: Question | None
    cost_usd: float
    problem: str | None = None


def companies() -> dict[str, str]:
    """doc_id → the company's registered name, for the prompt that must name it."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {doc_id: entry["company"] for doc_id, entry in manifest["documents"].items()}


def eligible(chunk: Chunk) -> bool:
    """Long enough to hold a fact, and prose rather than a table."""
    if len(chunk.text) < PASSAGE_MIN_CHARS:
        return False
    letters = sum(c.isalpha() or c.isspace() for c in chunk.text)
    return letters / len(chunk.text) >= PASSAGE_MIN_ALPHA


def sample_passages(inventory: Inventory, n: int) -> list[Chunk]:
    """`n` eligible passages spread evenly through the corpus, the same ones on every run.

    Evenly spaced rather than random: a question set that changes between runs cannot be compared
    with the score it produced last time, and a random sample lands mostly in whichever report is
    longest.
    """
    passages = [chunk for chunk in inventory.chunks if eligible(chunk)]
    if n <= 0:
        return []
    if n >= len(passages):
        return passages
    if n == 1:
        return passages[:1]
    step = (len(passages) - 1) / (n - 1)
    return [passages[round(i * step)] for i in range(n)]


def prompt_for(chunk: Chunk, company: str) -> list[Msg]:
    """The one user message: which company this is, and the passage itself."""
    return [{"role": "user", "content": f"company: {company}\n\npassage:\n{chunk.text}"}]


def verify(question: Question, document: Document, others: Sequence[Document] = ()) -> str | None:
    """Why this question cannot be used, or None if it can.

    The quote has to be findable in the document it came from and nowhere else in `others`.
    `squeeze` forgives whitespace, case, curly quotes and the spacing PDF extraction leaves behind;
    it forgives nothing else, so a paraphrase is caught here rather than scoring zero later.
    """
    if not question.usable:
        return "model judged the passage unusable"
    if not question.question.strip() or not question.quote.strip():
        return "question or quote is empty"
    quote = squeeze(question.quote)
    if len(quote) < 25:
        return f"quote is too short to identify a passage: {question.quote!r}"
    if quote not in squeeze(document.text):
        return f"quote is not in the document verbatim: {question.quote!r}"
    if quote in squeeze(question.question):
        return "the question contains its own answer"
    elsewhere = [other.doc_id for other in others if quote in squeeze(other.text)]
    if elsewhere:
        return f"quote also appears in {', '.join(sorted(elsewhere))}: {question.quote!r}"
    return None


def generate(
    llm: Llm, chunk: Chunk, company: str, model: str = MODEL
) -> tuple[Question | None, float]:
    """Ask for one question about one passage. Returns what came back and what it cost."""
    request = Request(
        chunk.id,
        prompt_for(chunk, company),
        MAX_TOKENS,
        system=SYSTEM,
        model=model,
        thinking=THINKING_EVAL_PARAM,
    )
    response = llm.parse(request, Question)
    return response.parsed, usage_cost(model, response.usage)


def to_row(index: int, chunk: Chunk, question: Question) -> dict[str, object]:
    """One verified question as a `Case`-shaped row, ready for the dataset."""
    return {
        "id": f"rq-{index:03d}",
        "input": {"question": question.question.strip()},
        "expected": {"quote": question.quote.strip()},
        "tags": [chunk.doc_id, question.topic.strip().casefold()],
    }


def write_dataset(rows: Sequence[dict[str, object]], path: Path = DATASET) -> Path:
    """Write the dataset, one row per line, in id order."""
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8"
    )
    return path


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aigent.retrieval.questions",
        description="Generate the retrieval question set from the corpus.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="send the calls; without it, count tokens only"
    )
    parser.add_argument(
        "--n", type=int, default=60, help="passages to ask about (some will not survive checking)"
    )
    parser.add_argument("--model", default=MODEL, help=f"which model writes them (default {MODEL})")
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    # pypdf warns once per font that it cannot fully decode; a dozen width arrays would bury
    # the cost line this command exists to print.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    documents = load_corpus()
    by_doc = {document.doc_id: document for document in documents}
    names = companies()
    inventory = Inventory.build("sentence", documents, by_sentence())
    passages = sample_passages(inventory, args.n)

    print(f"corpus: {len(documents)} documents, {len(inventory)} sentence chunks")
    print(f"sampling {len(passages)} passages on {args.model}")
    for doc_id in sorted({chunk.doc_id for chunk in passages}):
        share = sum(1 for chunk in passages if chunk.doc_id == doc_id)
        print(f"  {doc_id}: {share} passages")

    # Counting is free, so price the real request rather than guessing at its size.
    llm = Llm(budget=Budget(limit_usd=MAX_USD_PER_EVAL, scope="eval"))
    longest = max(passages, key=lambda chunk: len(chunk.text))
    probe = Request(
        longest.id,
        prompt_for(longest, names[longest.doc_id]),
        MAX_TOKENS,
        system=SYSTEM,
        model=args.model,
        thinking=THINKING_EVAL_PARAM,
    )
    tokens = llm.count(probe, Question)
    worst = estimate_eval_usd(args.model, len(passages), tokens, MAX_TOKENS)
    print(f"\n{tokens} input tokens for the largest passage, {MAX_TOKENS} output cap")
    print(f"worst case ${worst:.2f} against a ${MAX_USD_PER_EVAL:.2f} ceiling")

    if not args.yes:
        print("\nnothing spent. re-run with --yes to send these calls.")
        return
    llm.budget.admit(worst)

    results: list[Generated] = []
    for number, chunk in enumerate(passages, start=1):
        question, cost = generate(llm, chunk, names[chunk.doc_id], args.model)
        problem = (
            "no structured output"
            if question is None
            else verify(
                question,
                by_doc[chunk.doc_id],
                [doc for doc in documents if doc.doc_id != chunk.doc_id],
            )
        )
        results.append(Generated(chunk, question, cost, problem))
        print(f"[{number}/{len(passages)}] {chunk.id} {problem or 'ok'}  ${cost:.5f}")

    kept = [result for result in results if result.problem is None and result.question]
    rows = [
        to_row(index, result.chunk, result.question)
        for index, result in enumerate(kept, start=1)
        if result.question is not None
    ]
    path = write_dataset(rows)

    spent = sum(result.cost_usd for result in results)
    print(f"\n{len(rows)} of {len(passages)} survived checking, written to {path.name}")
    print(f"spent ${spent:.4f}")
    report_rejections(results)


def report_rejections(results: Sequence[Generated]) -> None:
    """Why questions were thrown away, which is a finding about the prompt or the corpus."""
    rejected = [result for result in results if result.problem is not None]
    if not rejected:
        print("every question passed checking")
        return
    print(f"\n{len(rejected)} rejected:")
    for result in rejected:
        print(f"  {result.chunk.id}: {result.problem}")


if __name__ == "__main__":
    main()
