"""Project 1's generation half: answer a question from retrieved passages.

uv run python -m entropic.retrieval.answer           # counts tokens, spends nothing
uv run python -m entropic.retrieval.answer --yes     # sends the calls

Two arms over the same 54 questions: `closed_book` asks the model cold, `rag` hands it the top k
chunks. The gap between them is what retrieval bought, and it is the number this module exists to
produce. Retrieval was graded first and separately (`retrieval.evaluate`), so a wrong answer on a
row where retrieval returned nothing is already known not to be the generator's fault.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import anthropic
from anthropic.types import MessageParam
from pydantic import BaseModel, Field, JsonValue

from entropic.config import (
    JUDGE_MODEL,
    MAX_TOKENS_JUDGE,
    MAX_USD_PER_EVAL,
    MODEL,
    THINKING_EVAL_PARAM,
    TOP_K,
    get_client,
)
from entropic.config import MAX_TOKENS_ANSWER as MAX_TOKENS
from entropic.evals.dataset import Case, digest, load_jsonl
from entropic.evals.grade import Grader, Outcome, flag, hit_at_k
from entropic.evals.judge import JUDGE_SYSTEM, LlmJudge, Verdict
from entropic.evals.report import write_report
from entropic.evals.runner import Task, run_eval
from entropic.pricing import check_request, estimate_eval_usd
from entropic.retrieval.chunk import Chunk, Inventory, context_block
from entropic.retrieval.corpus import load_corpus
from entropic.retrieval.dense import DenseIndex
from entropic.retrieval.embed import Embedder, LocalEmbedder, Vectors
from entropic.retrieval.evaluate import DATASET, STRATEGIES, resolve

REPO = Path(__file__).resolve().parents[3]

ANSWER_SYSTEM = """You answer questions about Indian listed companies from their annual reports.

You are given the question and a set of passages retrieved from those reports. Answer from the \
passages and from nothing else.

- Answer in one or two sentences. State the fact; do not restate the question.
- Cite the id of every passage you used, exactly as it is written in the `id` attribute.
- If the passages do not answer the question, set `answered` to false and leave `answer` empty. \
An honest miss is worth more than a plausible guess: a retrieval system that fabricates is worse \
than one that says it found nothing, because only one of the two can be checked.
- Do not use anything you know about these companies that is not in the passages. If a passage \
contradicts what you remember, the passage is right."""

CLOSED_BOOK_SYSTEM = """You answer questions about Indian listed companies from memory.

- Answer in one or two sentences. State the fact; do not restate the question.
- If you do not know the answer for the specific company and year asked about, set `answered` to \
false and leave `answer` empty. A wrong figure is worse than no figure.
- Leave `cited` empty: you have no passages to cite."""

RUBRIC = """The reference is the sentence from the company's annual report that answers the \
question. It is the ground truth.

Pass the answer only if it states the same fact as the reference — the same figure, name, date or \
commitment. Different wording is fine. Extra context is fine if it is correct and the reference \
supports it. A figure rounded differently is fine if it is the same figure.

Fail the answer if it states a different fact, if it adds a claim the reference does not support, \
or if it declines to answer. Declining is the right behaviour when the passages are not there, \
and it is still not a correct answer to this question."""


class Answer(BaseModel):
    """One answer and the passages it came from."""

    answered: bool = Field(description="False when the passages do not answer the question.")
    answer: str = Field(description="One or two sentences. Empty when `answered` is false.")
    cited: list[str] = Field(description="Ids of the passages used, copied exactly.")


@dataclass(frozen=True)
class Arm:
    """One arm of the eval: a system prompt, and whether it gets retrieved passages at all."""

    system: str
    retrieves: bool


ARMS = {
    "closed_book": Arm(CLOSED_BOOK_SYSTEM, retrieves=False),
    "rag": Arm(ANSWER_SYSTEM, retrieves=True),
}


def prompt_for(question: str, chunks: Sequence[Chunk]) -> list[MessageParam]:
    """The one user message. With no chunks this is the closed-book prompt."""
    if not chunks:
        return [{"role": "user", "content": f"question: {question}"}]
    content = f"{context_block(chunks)}\n\nquestion: {question}"
    return [{"role": "user", "content": content}]


def answer_task(
    arm: Arm,
    inventory: Inventory,
    index: DenseIndex,
    queries: Mapping[str, Vectors],
    k: int = TOP_K,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> Task:
    """Build the `Task` the runner calls once per case. `client` is injectable so tests are free."""
    built = client

    def task(case: Case) -> Outcome:
        nonlocal built
        if built is None:
            built = get_client()
        question = case.input.get("question")
        if not isinstance(question, str):
            return Outcome(error=f"case {case.id} has no question")

        chunks: list[Chunk] = []
        if arm.retrieves:
            vector = queries.get(case.id)
            if vector is None:
                return Outcome(error=f"case {case.id} has no query vector")
            chunks = [inventory.by_id[hit.chunk_id] for hit in index.search(vector, k=k)]

        messages = prompt_for(question, chunks)
        check_request(
            built,
            model=model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            system=arm.system,
            output_format=Answer,
        )
        response = built.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=arm.system,
            messages=messages,
            thinking=THINKING_EVAL_PARAM,
            output_format=Answer,
        )
        record = response.parsed_output
        if record is None:
            return Outcome(
                usage=response.usage,
                model=model,
                error=f"no parsed output; stop_reason={response.stop_reason}",
            )
        return Outcome(
            output={
                "answered": record.answered,
                "answer": record.answer,
                "cited": cast(JsonValue, record.cited),
            },
            # The judge reads `raw`, so it sees the answer and not the ids beside it.
            raw=record.answer if record.answered else "(declined to answer)",
            usage=response.usage,
            model=model,
        )

    return task


def graders(k: int = TOP_K, client: anthropic.Anthropic | None = None) -> dict[str, Grader]:
    """Two free graders and one paid one, read in that order: did it answer, did it cite a
    passage that holds the answer, was the answer right.

    `answered` is what keeps an abstention distinguishable from a confabulation, which `correct`
    alone cannot do — both fail it, and only one is safe. `cites_relevant` reuses `hit_at_k` over
    the *cited* ids rather than the retrieved ones, so a citation is checked against the label
    rather than trusted; it costs nothing and the judge could not make it, never seeing which chunk
    the label resolved to.
    """
    return {
        "answered": flag("answered"),
        "cites_relevant": hit_at_k(k, retrieved="cited"),
        "correct": LlmJudge(rubric=RUBRIC, reference="quote", client=client),
    }


def _judged_shape() -> Outcome:
    """An answer of about the length the cap allows, for counting the judge's prompt before a run.

    The judge's prompt is the rubric, the question, the reference and the answer; only the answer
    is unknown before the run, so a full-length stand-in prices the worst of it.
    """
    return Outcome(raw="word " * 60)


def spread(cases: Sequence[Case], n: int) -> list[Case]:
    """`n` cases spread across the set, for a cheap smoke run before the whole thing."""
    if n >= len(cases) or n <= 0:
        return list(cases)
    if n == 1:
        return list(cases[:1])
    step = (len(cases) - 1) / (n - 1)
    return [cases[round(i * step)] for i in range(n)]


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="entropic.retrieval.answer",
        description="Project 1: answer the question set from retrieved passages.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="send the calls; without it, estimate only"
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="run N cases spread across the dataset, for a cheap smoke run before the whole thing",
    )
    parser.add_argument("--k", type=int, default=TOP_K, help=f"passages given (default {TOP_K})")
    parser.add_argument(
        "--strategy",
        default="by_sentence",
        choices=sorted(STRATEGIES),
        help="which chunking strategy supplies the passages (default by_sentence)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="reuse extracted PDF text from corpus/.cache instead of re-reading the PDFs",
    )
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    cases = load_jsonl(DATASET)
    if args.sample is not None:
        cases = spread(cases, args.sample)

    documents = load_corpus(cache=args.cache)
    embedder: Embedder = LocalEmbedder()
    inventory = Inventory.build(args.strategy, documents, STRATEGIES[args.strategy])
    index = DenseIndex.build(inventory, embedder)
    resolved = resolve(cases, inventory)
    queries = {case.id: embedder.embed_query(str(case.input["question"])) for case in cases}

    sampled = f", sampled to {len(cases)}" if args.sample is not None else ""
    print(f"{DATASET.name}: {len(cases)} cases{sampled}")
    print(f"passages: {args.strategy}, {len(inventory):,} chunks, top {args.k} per question")
    print(f"arms: {', '.join(ARMS)}  on {MODEL}; judged by {JUDGE_MODEL}")

    def passages(case: Case) -> list[Chunk]:
        return [inventory.by_id[hit.chunk_id] for hit in index.search(queries[case.id], args.k)]

    # Counting is free, so price the real request rather than guessing at its size.
    client = get_client()
    worst = 0.0
    for name, arm in ARMS.items():
        tokens = max(
            client.messages.count_tokens(
                model=MODEL,
                system=arm.system,
                messages=prompt_for(
                    str(case.input["question"]), passages(case) if arm.retrieves else []
                ),
                output_format=Answer,
            ).input_tokens
            for case in cases
        )
        arm_usd = estimate_eval_usd(MODEL, len(cases), tokens, MAX_TOKENS)
        worst += arm_usd
        print(f"  {name}: {tokens} input tokens for the largest row, ${arm_usd:.2f} worst case")

    # The judge is half this eval's calls, so count its real prompt too rather than guessing.
    judge = graders(args.k, client=client)["correct"]
    assert isinstance(judge, LlmJudge)
    longest = max(resolved, key=lambda case: len(str(case.expected.get("quote", ""))))
    judge_tokens = client.messages.count_tokens(
        model=JUDGE_MODEL,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge.prompt_for(longest, _judged_shape())}],
        output_format=Verdict,
    ).input_tokens
    judged = estimate_eval_usd(JUDGE_MODEL, len(cases), judge_tokens, MAX_TOKENS_JUDGE, len(ARMS))
    worst += judged
    verdicts = len(cases) * len(ARMS)
    print(f"  judge: {verdicts} verdicts on {JUDGE_MODEL}, {judge_tokens} in, ${judged:.2f} worst")
    print(f"\nworst case ${worst:.2f} against a ${MAX_USD_PER_EVAL:.2f} ceiling")

    if not args.yes:
        print("\nnothing spent. re-run with --yes to send these calls.")
        return

    run = run_eval(
        resolved,
        {
            name: answer_task(arm, inventory, index, queries, args.k, client=client)
            for name, arm in ARMS.items()
        },
        graders(args.k, client=client),
        dataset=DATASET.name,
        digest=digest(DATASET),
        model=MODEL,
    )
    print(f"\nreport: {write_report(run)}")


if __name__ == "__main__":
    main()
