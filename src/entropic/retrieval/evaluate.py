"""The retrieval eval: one question set, three chunking strategies, four numbers each.

uv run python -m entropic.retrieval.evaluate

Nothing here calls the Anthropic API, so the whole run is free and can be repeated as often as the
question is worth asking. A label is a quote, resolved against each inventory in turn — which is
what lets three strategies be three columns of one table rather than three incomparable runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from entropic.config import CHUNK_CHARS, CHUNK_OVERLAP_CHARS, TOP_K
from entropic.evals.dataset import Case, digest, load_jsonl
from entropic.evals.grade import (
    Grader,
    Outcome,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    resolvable,
)
from entropic.evals.report import write_report
from entropic.evals.runner import Task, combine, run_eval
from entropic.retrieval.chunk import Inventory, Splitter, by_sentence, fixed
from entropic.retrieval.corpus import load_corpus
from entropic.retrieval.embed import Embedder, LocalEmbedder, Vectors
from entropic.retrieval.store import VectorStore, rank_ids

REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "evals" / "datasets" / "retrieval.jsonl"

# `by_heading` is deliberately absent: it resolves fewer labels than `by_sentence` while producing
# the most chunks of any strategy, which is heading detection firing on kerning damage. Kept in
# `chunk.py` to improve, not run as a column. See the knowledge graph.
STRATEGIES: dict[str, Splitter] = {
    "fixed": fixed(CHUNK_CHARS, 0),
    "fixed+overlap": fixed(CHUNK_CHARS, CHUNK_OVERLAP_CHARS),
    "by_sentence": by_sentence(),
}


def graders(k: int = TOP_K) -> dict[str, Grader]:
    """What each row is asked, widest question first.

    `resolvable` is the denominator of the three below it; `hit@k` is the one that decides whether
    an answer is possible at all, and `recall@k` and `mrr` refine it.
    """
    return {
        "resolvable": resolvable(),
        f"hit@{k}": hit_at_k(k),
        f"recall@{k}": recall_at_k(k),
        "mrr": reciprocal_rank(),
    }


def resolve(cases: Sequence[Case], inventory: Inventory) -> list[Case]:
    """Each case's labelled quote turned into the chunk ids that hold it in *this* inventory.

    An empty list is kept rather than dropped: it is what `resolvable` reports, and dropping the
    row would quietly raise every other strategy's score by removing the cases it fails.
    """
    resolved: list[Case] = []
    for case in cases:
        quote = case.expected.get("quote")
        ids = inventory.containing(quote) if isinstance(quote, str) else []
        resolved.append(
            Case(
                id=case.id,
                input=case.input,
                expected={**case.expected, "relevant": cast(JsonValue, ids)},
                tags=case.tags,
            )
        )
    return resolved


def retrieval_task(store: VectorStore, queries: Mapping[str, Vectors], k: int = TOP_K) -> Task:
    """Rank this store against a pre-embedded question. Spends nothing, so `usage` stays None."""

    def task(case: Case) -> Outcome:
        vector = queries.get(case.id)
        if vector is None:
            return Outcome(error=f"case {case.id} has no query vector")
        hits = store.search(vector, k=k)
        return Outcome(
            output={
                "retrieved": cast(JsonValue, rank_ids(hits)),
                "scores": cast(JsonValue, [hit.score for hit in hits]),
            }
        )

    return task


def report_unresolved(name: str, cases: Sequence[Case]) -> None:
    """Labels no single chunk holds. A finding about the splitter, not a broken dataset."""
    lost = [case.id for case in cases if not case.expected.get("relevant")]
    if not lost:
        return
    print(f"    {len(lost)} labels straddle a boundary under {name}: {', '.join(lost)}")


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="entropic.retrieval.evaluate",
        description="Score the retriever over the question set. Free: no API calls.",
    )
    parser.add_argument("--k", type=int, default=TOP_K, help=f"chunks retrieved (default {TOP_K})")
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
    started = time.perf_counter()
    documents = load_corpus(cache=args.cache)
    print(f"{DATASET.name}: {len(cases)} cases over {len(documents)} documents")

    embedder: Embedder = LocalEmbedder()
    print(f"embedder: {embedder.name}, {embedder.dimensions} dimensions")
    queries = {case.id: embedder.embed_query(str(case.input["question"])) for case in cases}

    runs = []
    for name, splitter in STRATEGIES.items():
        inventory = Inventory.build(name, documents, splitter)
        store = VectorStore.build(inventory, embedder)
        resolved = resolve(cases, inventory)
        truncated = (
            embedder.count_truncated(inventory.texts())
            if isinstance(embedder, LocalEmbedder)
            else 0
        )
        print(f"  {name}: {len(inventory):,} chunks, {truncated} past the token limit")
        report_unresolved(name, resolved)
        runs.append(
            run_eval(
                resolved,
                {name: retrieval_task(store, queries, args.k)},
                graders(args.k),
                dataset=DATASET.name,
                digest=digest(DATASET),
                model=embedder.name,
                progress=False,
            )
        )

    run = combine(runs)
    print(f"\n{len(run.rows)} rows in {time.perf_counter() - started:.1f}s, $0.00 spent\n")
    print(f"report: {write_report(run)}")


if __name__ == "__main__":
    main()
