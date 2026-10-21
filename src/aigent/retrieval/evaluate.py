"""The retrieval eval: one question set, two comparisons, four numbers each.

uv run python -m aigent.retrieval.evaluate                      # four chunking strategies
uv run python -m aigent.retrieval.evaluate --compare retrieval  # six retrieval methods

The prose questions and the table questions live in two files and are scored as one set. Each
question is searched within the one report its tags name, as the analyst searches.

Nothing here calls the Anthropic API, so both runs are free and can be repeated as often as the
question is worth asking.

The two comparisons differ in a way worth knowing before reading either table. Chunking strategies
each build their own inventory, where the same chunk id names different text, so their *labels*
differ and they are separate runs merged by `combine`. Retrieval methods all rank one inventory, so
they share their labels and are six tasks of a single run. A label is a quote either way, resolved
against the inventory being scored, which is what makes any of these columns comparable at all.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from aigent.config import (
    CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    RERANK_CANDIDATES,
    SEARCH_METHOD,
    TOP_K,
)
from aigent.evals.dataset import Case, DatasetError, digest, load_jsonl
from aigent.evals.grade import (
    Grader,
    Outcome,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    resolvable,
)
from aigent.evals.report import write_report
from aigent.evals.runner import EvalRun, Task, combine, run_eval
from aigent.retrieval.chunk import Document, Inventory, Splitter, by_element, by_sentence, fixed
from aigent.retrieval.corpus import FY25_REPORTS, load_corpus
from aigent.retrieval.dense import DenseIndex
from aigent.retrieval.embed import Embedder, LocalEmbedder, Vectors
from aigent.retrieval.hits import rank_ids
from aigent.retrieval.keys import keyed_search
from aigent.retrieval.rank import METHODS, Indexes, Ranker, build_ranker, scoped
from aigent.retrieval.rerank import LocalReranker

REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "evals" / "datasets" / "retrieval.jsonl"
TABLES = DATASET.with_name("retrieval-tables.jsonl")
DATASETS = (DATASET, TABLES)

# `by_heading` is deliberately absent: it resolves fewer labels than `by_sentence` while producing
# the most chunks of any strategy, which is heading detection firing on kerning damage. Kept in
# `chunk.py` to improve, not run as a column. See the knowledge graph.
STRATEGIES: dict[str, Splitter] = {
    "fixed": fixed(CHUNK_CHARS, 0),
    "fixed+overlap": fixed(CHUNK_CHARS, CHUNK_OVERLAP_CHARS),
    "by_sentence": by_sentence(),
    "by_element": by_element(),
}

# The inventory the six retrieval methods all rank, so they share one set of labels.
METHOD_STRATEGY = "by_element"


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


def load_cases(paths: Sequence[Path]) -> list[Case]:
    """Every file's cases as one set; an id two files share is a `DatasetError`."""
    cases: list[Case] = []
    seen: dict[str, str] = {}
    for path in paths:
        for case in load_jsonl(path):
            if case.id in seen:
                raise DatasetError(f"{path.name}: id {case.id!r} is also in {seen[case.id]}")
            seen[case.id] = path.name
            cases.append(case)
    return cases


def label_of(paths: Sequence[Path]) -> tuple[str, str]:
    """The name and digest a report records: one file's own, or the files' joined."""
    if len(paths) == 1:
        return paths[0].name, digest(paths[0])
    joined = "".join(digest(path) for path in paths).encode()
    return "+".join(path.name for path in paths), hashlib.sha256(joined).hexdigest()[:12]


def report_of(case: Case) -> str | None:
    """The one report a case's tags name; None when they name none or several."""
    named = [tag for tag in case.tags if tag in FY25_REPORTS]
    return named[0] if len(named) == 1 else None


def _unscoped(case: Case) -> Outcome:
    return Outcome(error=f"case {case.id} names no single report to search")


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


def retrieval_task(
    index: DenseIndex, queries: Mapping[str, Vectors], inventory: Inventory, k: int = TOP_K
) -> Task:
    """Rank the case's report in this index against a pre-embedded question. Spends nothing."""

    def task(case: Case) -> Outcome:
        vector = queries.get(case.id)
        if vector is None:
            return Outcome(error=f"case {case.id} has no query vector")
        report = report_of(case)
        if report is None:
            return _unscoped(case)
        hits = scoped(lambda n: index.search(vector, k=n), k, report, inventory)
        return Outcome(
            output={
                "retrieved": cast(JsonValue, rank_ids(hits)),
                "scores": cast(JsonValue, [hit.score for hit in hits]),
            }
        )

    return task


def ranker_task(ranker: Ranker, k: int = TOP_K) -> Task:
    """Rank the question within its case's report with this ranker. Spends nothing."""

    def task(case: Case) -> Outcome:
        report = report_of(case)
        if report is None:
            return _unscoped(case)
        hits = ranker.rank(str(case.input["question"]), k, doc_id=report)
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
        prog="aigent.retrieval.evaluate",
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
        nargs="+",
        default=list(DATASETS),
        help="question sets scored as one (default the prose and the table questions)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="reuse PDF text and chunk vectors from corpus/.cache instead of recomputing them",
    )
    return parser.parse_args(list(argv))


def compare_chunking(
    cases: Sequence[Case],
    documents: Sequence[Document],
    embedder: Embedder,
    k: int,
    datasets: Sequence[Path] = DATASETS,
    candidates: int = RERANK_CANDIDATES,
    cache: bool = False,
) -> EvalRun:
    """One run per strategy, merged. Separate runs because the *labels* differ per inventory:
    a chunk id names different text under each splitter, so one `run_eval` cannot serve them all.

    `candidates` is accepted and unused: nothing here reranks, and the two comparisons are called
    through one signature.
    """
    del candidates
    dataset, fingerprint = label_of(datasets)
    queries = {case.id: embedder.embed_query(str(case.input["question"])) for case in cases}
    runs = []
    for name, splitter in STRATEGIES.items():
        inventory = Inventory.build(name, documents, splitter)
        index = (DenseIndex.build_cached if cache else DenseIndex.build)(inventory, embedder)
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
                {name: retrieval_task(index, queries, inventory, k)},
                graders(k),
                dataset=dataset,
                digest=fingerprint,
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
    datasets: Sequence[Path] = DATASETS,
    candidates: int = RERANK_CANDIDATES,
    cache: bool = False,
) -> EvalRun:
    """Six methods over one inventory, and the keyed search production runs over the same chunks
    (`keyed_search`), so one run with seven tasks and no merge.

    Every arm ranks the same chunks with the same ids, which means the same resolved labels and
    the same `resolvable` column — so the differences between these columns are the retrieval
    methods and nothing else.
    """
    dataset, fingerprint = label_of(datasets)
    inventory = Inventory.build(METHOD_STRATEGY, documents, STRATEGIES[METHOD_STRATEGY])
    indexes = Indexes.build(inventory, embedder, cache=cache)
    reranker = LocalReranker()
    rankers = {
        method: build_ranker(method, indexes, reranker, candidates=candidates) for method in METHODS
    }
    _, rankers[f"keys+{SEARCH_METHOD}"] = keyed_search(
        METHOD_STRATEGY,
        documents,
        cache=cache,
        embedder=embedder,
        reranker=reranker,
        candidates=candidates,
    )
    resolved = resolve(cases, inventory)
    print(f"  chunks: {len(inventory):,} under {METHOD_STRATEGY}")
    print(f"  lexical: {len(indexes.sparse.idf):,} distinct terms")
    print(f"  reranker: {reranker.name}, top {candidates} rescored to {k}")
    report_unresolved(METHOD_STRATEGY, resolved)
    return run_eval(
        resolved,
        {method: ranker_task(ranker, k) for method, ranker in rankers.items()},
        graders(k),
        dataset=dataset,
        digest=fingerprint,
        model=embedder.name,
        progress=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    cases = load_cases(args.dataset)
    started = time.perf_counter()
    documents = load_corpus(cache=args.cache, doc_ids=FY25_REPORTS)
    print(f"{label_of(args.dataset)[0]}: {len(cases)} cases over {len(documents)} documents")

    embedder: Embedder = LocalEmbedder()
    print(f"embedder: {embedder.name}, {embedder.dimensions} dimensions")
    print(f"comparing: {args.compare}")

    compare = compare_chunking if args.compare == "chunking" else compare_retrieval
    run = compare(cases, documents, embedder, args.k, args.dataset, args.candidates, args.cache)

    print(f"\n{len(run.rows)} rows in {time.perf_counter() - started:.1f}s, $0.00 spent\n")
    print(f"report: {write_report(run)}")


if __name__ == "__main__":
    main()
