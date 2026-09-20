"""The retrieval eval: one question set, two comparisons, four numbers each.

uv run python -m entropic.retrieval.evaluate                      # three chunking strategies
uv run python -m entropic.retrieval.evaluate --compare retrieval  # four retrieval methods

Nothing here calls the Anthropic API, so both runs are free and can be repeated as often as the
question is worth asking.

The two comparisons differ in a way worth knowing before reading either table. Chunking strategies
each build their own inventory, where the same chunk id names different text, so their *labels*
differ and they are separate runs merged by `combine`. Retrieval methods all rank one inventory, so
they share their labels and are four tasks of a single run. A label is a quote either way, resolved
against the inventory being scored, which is what makes any of these columns comparable at all.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from entropic.config import (
    CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    FUSE_DEPTH,
    RERANK_CANDIDATES,
    TOP_K,
)
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
from entropic.evals.runner import EvalRun, Task, combine, run_eval
from entropic.retrieval.chunk import Document, Inventory, Splitter, by_sentence, fixed
from entropic.retrieval.corpus import load_corpus
from entropic.retrieval.embed import Embedder, LocalEmbedder, Vectors
from entropic.retrieval.fuse import reciprocal_rank_fusion
from entropic.retrieval.rerank import LocalReranker, Reranker, rerank
from entropic.retrieval.sparse import Bm25Index
from entropic.retrieval.store import Hit, VectorStore, rank_ids

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


# The second comparison this module runs, and a different axis from the one above: these rank the
# *same* `by_sentence` inventory, so they share a set of labels and are tasks of one run rather
# than runs merged. Three bases, each with and without a cross-encoder over its own candidates,
# so the grid separates what reach buys from what ordering buys.
METHODS = (
    "dense",
    "dense+rerank",
    "bm25",
    "bm25+rerank",
    "hybrid",
    "hybrid+rerank",
)
METHOD_STRATEGY = "by_sentence"


@dataclass(frozen=True)
class Retrievers:
    """Everything the four methods rank with, built once over one inventory.

    `reranker` is None until an arm needs one: it is a second local model to download and load,
    and the three cheap arms must not pay for it.
    """

    inventory: Inventory
    store: VectorStore
    index: Bm25Index
    reranker: Reranker | None = None


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


def base_ranking(
    base: str,
    retrievers: Retrievers,
    question: str,
    vector: Vectors,
    k: int,
    depth: int,
) -> list[Hit]:
    """The cheap first stage: one of dense, lexical, or the two fused.

    `bm25` reads the question text and `dense` reads its vector, which is why both are carried —
    a lexical ranker cannot use an embedding and a dense one cannot use the words.
    """
    if base == "bm25":
        return retrievers.index.search(question, k=k)
    if base == "dense":
        return retrievers.store.search(vector, k=k)
    # Fuse the top `depth` of each ranking rather than its head: a chunk at rank 20 lexically and
    # rank 4 densely is what fusion is for, and cutting at k throws it away unseen.
    return reciprocal_rank_fusion(
        [
            rank_ids(retrievers.store.search(vector, k=depth)),
            rank_ids(retrievers.index.search(question, k=depth)),
        ],
        k=k,
    )


def method_task(
    method: str,
    retrievers: Retrievers,
    queries: Mapping[str, Vectors],
    k: int = TOP_K,
    *,
    depth: int = FUSE_DEPTH,
    candidates: int = RERANK_CANDIDATES,
) -> Task:
    """Rank one inventory by one retrieval method. Spends nothing, so `usage` stays None.

    A `+rerank` arm is its base with a second stage: take `candidates` from the base instead of
    `k`, score every one against the query with the cross-encoder, keep the best `k`. The base is
    what decides reach; the reranker only decides order within what the base reached.
    """
    if method not in METHODS:
        raise ValueError(f"unknown retrieval method {method!r}; expected one of {METHODS}")
    base, reranked = method.removesuffix("+rerank"), method.endswith("+rerank")

    def task(case: Case) -> Outcome:
        question = str(case.input["question"])
        vector = queries.get(case.id)
        if vector is None:
            return Outcome(error=f"case {case.id} has no query vector")

        hits = base_ranking(
            base, retrievers, question, vector, candidates if reranked else k, depth
        )
        if reranked:
            if retrievers.reranker is None:
                return Outcome(error="this arm needs a reranker and was given none")
            texts = [retrievers.inventory.by_id[hit.chunk_id].text for hit in hits]
            hits = rerank(retrievers.reranker, question, hits, texts, k=k)

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
    parser.add_argument(
        "--compare",
        choices=("chunking", "retrieval"),
        default="chunking",
        help="chunking strategies (default), or retrieval methods over one inventory",
    )
    parser.add_argument("--k", type=int, default=TOP_K, help=f"chunks retrieved (default {TOP_K})")
    parser.add_argument(
        "--candidates",
        type=int,
        default=RERANK_CANDIDATES,
        help=f"how many the reranker rescores (default {RERANK_CANDIDATES})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="question set to score against (default evals/datasets/retrieval.jsonl)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="reuse extracted PDF text from corpus/.cache instead of re-reading the PDFs",
    )
    return parser.parse_args(list(argv))


def compare_chunking(
    cases: Sequence[Case],
    documents: Sequence[Document],
    embedder: Embedder,
    k: int,
    dataset: Path = DATASET,
    candidates: int = RERANK_CANDIDATES,
) -> EvalRun:
    """One run per strategy, merged. Separate runs because the *labels* differ per inventory:
    a chunk id names different text under each splitter, so one `run_eval` cannot serve them all.

    `candidates` is accepted and unused: nothing here reranks, and the two comparisons are called
    through one signature.
    """
    del candidates
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
                {name: retrieval_task(store, queries, k)},
                graders(k),
                dataset=dataset.name,
                digest=digest(dataset),
                model=embedder.name,
                progress=False,
            )
        )
    return combine(runs)


def compare_retrieval(
    cases: Sequence[Case],
    documents: Sequence[Document],
    embedder: Embedder,
    k: int,
    dataset: Path = DATASET,
    candidates: int = RERANK_CANDIDATES,
) -> EvalRun:
    """Four methods over one inventory, so one run with four tasks and no merge.

    Every arm ranks the same chunks with the same ids, which means the same resolved labels and
    the same `resolvable` column — so the differences between these columns are the retrieval
    methods and nothing else.
    """
    queries = {case.id: embedder.embed_query(str(case.input["question"])) for case in cases}
    inventory = Inventory.build(METHOD_STRATEGY, documents, STRATEGIES[METHOD_STRATEGY])
    reranker = LocalReranker()
    retrievers = Retrievers(
        inventory=inventory,
        store=VectorStore.build(inventory, embedder),
        index=Bm25Index.build(inventory),
        reranker=reranker,
    )
    resolved = resolve(cases, inventory)
    print(f"  chunks: {len(inventory):,} under {METHOD_STRATEGY}")
    print(f"  lexical: {len(retrievers.index.idf):,} distinct terms")
    print(f"  reranker: {reranker.name}, top {candidates} rescored to {k}")
    report_unresolved(METHOD_STRATEGY, resolved)
    return run_eval(
        resolved,
        {
            name: method_task(name, retrievers, queries, k, candidates=candidates)
            for name in METHODS
        },
        graders(k),
        dataset=dataset.name,
        digest=digest(dataset),
        model=embedder.name,
        progress=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    cases = load_jsonl(args.dataset)
    started = time.perf_counter()
    documents = load_corpus(cache=args.cache)
    print(f"{args.dataset.name}: {len(cases)} cases over {len(documents)} documents")

    embedder: Embedder = LocalEmbedder()
    print(f"embedder: {embedder.name}, {embedder.dimensions} dimensions")
    print(f"comparing: {args.compare}")

    compare = compare_chunking if args.compare == "chunking" else compare_retrieval
    run = compare(cases, documents, embedder, args.k, args.dataset, args.candidates)

    print(f"\n{len(run.rows)} rows in {time.perf_counter() - started:.1f}s, $0.00 spent\n")
    print(f"report: {write_report(run)}")


if __name__ == "__main__":
    main()
