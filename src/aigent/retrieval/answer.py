"""Project 1's generation half: answer a question from retrieved passages.

uv run python -m aigent.retrieval.answer           # counts tokens, spends nothing
uv run python -m aigent.retrieval.answer --yes     # sends the calls

Two arms over the same 54 questions: `closed_book` asks the model cold, `rag` hands it the top k
chunks. The gap between them is what retrieval bought, and it is the number this module exists to
produce. Retrieval was graded first and separately (`retrieval.evaluate`), so a wrong answer on a
row where retrieval returned nothing is already known not to be the generator's fault.

`--client` picks who answers and `--judge-client` who grades, so a model on this machine can be
judged by a hosted one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, JsonValue

from aigent.adapters import CLIENTS, judge_of, spec
from aigent.config import CLIENT, MAX_USD_PER_EVAL, THINKING_EVAL_PARAM, TOP_K, max_tokens
from aigent.evals.dataset import Case, digest, load_jsonl
from aigent.evals.grade import Grader, Outcome, flag, hit_at_k, unparsed
from aigent.evals.judge import JUDGE_SYSTEM, LlmJudge, Verdict
from aigent.evals.report import write_report
from aigent.evals.runner import Task, run_eval
from aigent.llm import Llm, Request
from aigent.messages import Msg
from aigent.pricing import estimate_eval_usd
from aigent.retrieval.chunk import Chunk, Inventory, context_block
from aigent.retrieval.corpus import load_corpus
from aigent.retrieval.dense import DenseIndex
from aigent.retrieval.embed import Embedder, LocalEmbedder, Vectors
from aigent.retrieval.evaluate import DATASET, STRATEGIES, resolve

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


def prompt_for(question: str, chunks: Sequence[Chunk]) -> list[Msg]:
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
    client: str = CLIENT,
    sdk: object | None = None,
    model: str | None = None,
) -> Task:
    """Build the `Task` the runner calls once per case. `sdk` is injectable, so tests are free.
    `model` and the output cap are this client's own rather than the configured client's — the two
    differ the moment an eval is pointed at a local server."""
    llm = Llm.for_eval(client, sdk=sdk)
    settings = spec(client)
    model = model or settings.model
    cap = max_tokens("ANSWER", settings)

    def task(case: Case) -> Outcome:
        question = case.input.get("question")
        if not isinstance(question, str):
            return Outcome(error=f"case {case.id} has no question")

        chunks: list[Chunk] = []
        if arm.retrieves:
            vector = queries.get(case.id)
            if vector is None:
                return Outcome(error=f"case {case.id} has no query vector")
            chunks = [inventory.by_id[hit.chunk_id] for hit in index.search(vector, k=k)]

        request = Request(
            case.id,
            prompt_for(question, chunks),
            cap,
            system=arm.system,
            model=model,
            thinking=THINKING_EVAL_PARAM,
        )
        response = llm.parse(request, Answer)
        record = response.parsed
        if record is None:
            return Outcome(
                usage=response.usage,
                model=model,
                error=unparsed(response.stop_reason, response.text),
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


def graders(k: int = TOP_K, client: str = CLIENT, sdk: object | None = None) -> dict[str, Grader]:
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
        "correct": LlmJudge(rubric=RUBRIC, reference="quote", client=client, sdk=sdk),
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
        prog="aigent.retrieval.answer",
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
        help="reuse PDF text and chunk vectors from corpus/.cache instead of recomputing them",
    )
    parser.add_argument(
        "--client", default=CLIENT, choices=sorted(CLIENTS), help=f"who answers (default {CLIENT})"
    )
    parser.add_argument(
        "--judge-client",
        choices=sorted(CLIENTS),
        help="who grades the answers (default: the one the answering client names, else itself)",
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
    index = (DenseIndex.build_cached if args.cache else DenseIndex.build)(inventory, embedder)
    resolved = resolve(cases, inventory)
    queries = {case.id: embedder.embed_query(str(case.input["question"])) for case in cases}

    # One `Llm` per client, so the tasks share a connection and the judge shares it or has its own.
    judge_client = args.judge_client or judge_of(args.client)
    answering = Llm.for_eval(args.client)
    judging = answering if judge_client == args.client else Llm.for_eval(judge_client)
    answerer, grader = spec(answering.client), spec(judging.client)
    answer_cap, judge_cap = max_tokens("ANSWER", answerer), max_tokens("JUDGE", grader)

    sampled = f", sampled to {len(cases)}" if args.sample is not None else ""
    print(f"{DATASET.name}: {len(cases)} cases{sampled}")
    print(f"passages: {args.strategy}, {len(inventory):,} chunks, top {args.k} per question")
    print(f"arms: {', '.join(ARMS)}  answered by {answerer.model} ({answerer.name})")
    print(f"judged by {grader.judge_model} ({grader.name})")

    def passages(case: Case) -> list[Chunk]:
        return [inventory.by_id[hit.chunk_id] for hit in index.search(queries[case.id], args.k)]

    # Counting is free, so price the real request rather than guessing at its size.
    worst = 0.0
    for name, arm in ARMS.items():
        tokens = max(
            answering.count(
                Request(
                    case.id,
                    prompt_for(
                        str(case.input["question"]), passages(case) if arm.retrieves else []
                    ),
                    answer_cap,
                    system=arm.system,
                    thinking=THINKING_EVAL_PARAM,
                ),
                Answer,
            )
            for case in cases
        )
        arm_usd = estimate_eval_usd(answerer.model, len(cases), tokens, answer_cap)
        worst += arm_usd
        print(f"  {name}: {tokens} input tokens for the largest row, ${arm_usd:.2f} worst case")

    # The judge is half this eval's calls, so count its real prompt too rather than guessing.
    judge = graders(args.k, client=judging.client, sdk=judging.sdk)["correct"]
    assert isinstance(judge, LlmJudge)
    longest = max(resolved, key=lambda case: len(str(case.expected.get("quote", ""))))
    prompt = judge.prompt_for(longest, _judged_shape())
    judged_by = grader.judge_model
    judged_row = Request.ask("judge", JUDGE_SYSTEM, prompt, judge_cap, model=judged_by)
    judge_tokens = judging.count(judged_row, Verdict)
    judged = estimate_eval_usd(judged_by, len(cases), judge_tokens, judge_cap, len(ARMS))
    worst += judged
    verdicts = len(cases) * len(ARMS)
    print(f"  judge: {verdicts} verdicts on {judged_by}, {judge_tokens} in, ${judged:.2f} worst")
    print(f"\nworst case ${worst:.2f} against a ${MAX_USD_PER_EVAL:.2f} ceiling")

    if not args.yes:
        print("\nnothing spent. re-run with --yes to send these calls.")
        return

    run = run_eval(
        resolved,
        {
            name: answer_task(
                arm, inventory, index, queries, args.k, client=answering.client, sdk=answering.sdk
            )
            for name, arm in ARMS.items()
        },
        graders(args.k, client=judging.client, sdk=judging.sdk),
        dataset=DATASET.name,
        digest=digest(DATASET),
        model=answerer.model,
        worst_usd=worst,
    )
    print(f"\nreport: {write_report(run)}")


if __name__ == "__main__":
    main()
