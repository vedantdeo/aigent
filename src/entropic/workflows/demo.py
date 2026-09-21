"""The command line the five patterns share: a dry run by default, `--yes` to spend."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence

from entropic.config import MAX_USD_PER_WORKFLOW, SEARCH_METHOD
from entropic.llm import Llm, Rehearsed, StepFailed, describe
from entropic.pricing import Budget, BudgetExceeded
from entropic.retrieval.chunk import Chunk, Inventory, by_sentence
from entropic.retrieval.corpus import load_corpus
from entropic.retrieval.rank import Indexes, build_ranker
from entropic.workflows.reports import Search


def build_search(cache: bool) -> Search:
    """Chunk, embed and index the corpus once; passages come back by `SEARCH_METHOD`."""
    inventory = Inventory.build("by_sentence", load_corpus(cache=cache), by_sentence())
    ranker = build_ranker(SEARCH_METHOD, Indexes.build(inventory))

    def search(query: str, /, *, doc_id: str | None = None) -> list[Chunk]:
        return [inventory.by_id[hit.chunk_id] for hit in ranker.rank(query, doc_id=doc_id)]

    return search


def run_demo[Result](
    argv: Sequence[str] | None,
    *,
    prog: str,
    questions: Sequence[str],
    run: Callable[[Llm, Search, str], Result],
    show: Callable[[Result], str],
) -> None:
    """Build the search, then ask each question through `run` and print what `show` makes of it.

    Without `--yes` the first model call is counted, priced and not sent.
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("question", nargs="?", help="ask this instead of the built-in examples")
    parser.add_argument("--yes", action="store_true", help="send the calls (default: dry run)")
    parser.add_argument(
        "--limit",
        type=float,
        default=MAX_USD_PER_WORKFLOW,
        help=f"hard ceiling in USD, checked before every call (default {MAX_USD_PER_WORKFLOW})",
    )
    parser.add_argument("--cache", action="store_true", help="reuse PDF text from corpus/.cache")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    started = time.perf_counter()
    search = build_search(args.cache)
    print(f"search built by {SEARCH_METHOD} in {time.perf_counter() - started:.0f}s\n")

    budget = Budget(limit_usd=args.limit, scope="workflow")
    llm = Llm(budget=budget, rehearse=not args.yes)
    for question in [args.question] if args.question else questions:
        print(f"question: {question}")
        started = time.perf_counter()
        try:
            result = run(llm, search, question)
        except Rehearsed as first:
            print(
                f"\ndry run: the first call ({first.request.step}, {first.request.model}) is "
                f"{first.input_tokens:,} input tokens, up to ${first.worst_usd:.4f}.\n"
                f"every call is admitted against a ${args.limit:.2f} ceiling before it is sent. "
                "--yes to spend."
            )
            return
        except (BudgetExceeded, StepFailed) as failed:
            print(describe(llm.trace))
            raise SystemExit(f"stopped: {failed}") from failed
        print(f"\n{show(result)}\n({time.perf_counter() - started:.1f}s)\n")
    print(describe(llm.trace))
