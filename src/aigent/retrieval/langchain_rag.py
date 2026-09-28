"""The same retrieval pipeline, rebuilt in LangChain, scored by the same harness.

uv sync --group compare
uv run python -m aigent.retrieval.langchain_rag

Week 2's framework comparison. The point is not that one is better — it is to find out what the
framework does *for* you and what it does *to* you, with a number attached rather than an opinion.
Same three PDFs, same 54 quote-labelled questions, same graders, same report: only the machinery
between corpus and ranked ids changes.

LangChain is imported inside the functions that need it, and its packages live in the `compare`
dependency group, so nothing here is installed for an ordinary `uv sync` and the core package never
depends on a framework it is being compared against.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import JsonValue

from aigent.config import CHUNK_CHARS, CHUNK_OVERLAP_CHARS, EMBED_MODEL, TOP_K
from aigent.evals.dataset import Case, digest, load_jsonl
from aigent.evals.grade import Outcome
from aigent.evals.report import write_report
from aigent.evals.runner import Task, run_eval
from aigent.retrieval.chunk import Chunk, Inventory
from aigent.retrieval.corpus import CORPUS_DIR
from aigent.retrieval.evaluate import DATASET, graders, resolve

if TYPE_CHECKING:
    from langchain_core.documents import Document as LcDocument

REPO = Path(__file__).resolve().parents[3]


class Retrieves(Protocol):
    """The one method this module needs from a LangChain retriever.

    A protocol for the reason `Embedder` and `Reranker` are protocols: the narrow surface is the
    dependency, so a test substitutes a fake without constructing a `VectorStoreRetriever` — which
    is a concrete Pydantic model and cannot be stood in for. Positional-only, so the parameter name
    LangChain chose is not part of the contract.
    """

    def invoke(self, query: str, /) -> list[LcDocument]: ...


def load_and_split(
    directory: Path = CORPUS_DIR,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[LcDocument]:
    """LangChain's loader and splitter, at our chunk size.

    `add_start_index=True` is **opt-in and off by default**, and without it a chunk cannot say
    where in its document it came from — which is what a page citation resolves through.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages: list[LcDocument] = []
    for path in sorted(directory.glob("*.pdf")):
        pages.extend(PyPDFLoader(str(path)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    return splitter.split_documents(pages)


def prepare(chunks: Sequence[LcDocument], name: str = "langchain") -> Inventory:
    """Give every chunk its id — written into metadata *and* used for our `Inventory` — in one pass.

    Both sides of the comparison need the same key, and they need it for different reasons. The
    `Inventory` is what `resolve` turns a labelled quote into chunk ids against; the metadata is
    what a retrieved chunk is recognised by, because a LangChain `Document` has no identity of its
    own and `InMemoryVectorStore` hands back reconstructed objects — matching on `id(chunk)` finds
    nothing, silently, and costs a run that reports 0.000 on every metric without raising.

    So the id is computed **once, here**. Deriving it twice is the same failure waiting to happen
    from the other direction: the labels and the retrieved ids would come from separate formulas,
    and nothing downstream can tell that apart from a retriever that simply missed.

    Mutates `chunks` in place, and must run before anything indexes them.
    """
    ours: list[Chunk] = []
    for ordinal, chunk in enumerate(chunks):
        source = Path(str(chunk.metadata.get("source", ""))).stem
        chunk_id = f"{source}#{ordinal:04d}"
        chunk.metadata["chunk_id"] = chunk_id
        page = chunk.metadata.get("page")
        ours.append(
            Chunk(
                id=chunk_id,
                text=chunk.page_content,
                doc_id=source,
                ordinal=ordinal,
                start=int(cast(int, chunk.metadata.get("start_index", 0))),
                # LangChain's loader 0-indexes pages; every citation in this repo is 1-indexed.
                page=None if page is None else int(cast(int, page)) + 1,
                heading=None,
            )
        )
    return Inventory(name=name, chunks=tuple(ours))


def build_retriever(chunks: Sequence[LcDocument], k: int = TOP_K) -> Retrieves:
    """LangChain's embeddings and in-memory store, on the same model we use."""
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
    )
    store = InMemoryVectorStore.from_documents(list(chunks), embeddings)
    return store.as_retriever(search_kwargs={"k": k})


def langchain_task(retriever: Retrieves) -> Task:
    """Rank through LangChain's retriever, answering in the ids our graders read.

    Reads the id back out of metadata, which `prepare` put there. A chunk that comes back without
    one is an error rather than a skipped row: silently dropping it is what turned a broken mapping
    into a report of zeros.
    """

    def task(case: Case) -> Outcome:
        question = str(case.input["question"])
        found = retriever.invoke(question)
        ids = [str(chunk.metadata["chunk_id"]) for chunk in found if "chunk_id" in chunk.metadata]
        if len(ids) != len(found):
            return Outcome(error=f"{len(found) - len(ids)} retrieved chunks carried no chunk_id")
        return Outcome(output={"retrieved": cast(JsonValue, ids)})

    return task


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aigent.retrieval.langchain_rag",
        description="The same retrieval pipeline in LangChain. Free: no API calls.",
    )
    parser.add_argument("--k", type=int, default=TOP_K, help=f"chunks retrieved (default {TOP_K})")
    parser.add_argument(
        "--dataset", type=Path, default=DATASET, help="question set to score against"
    )
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    cases = load_jsonl(args.dataset)
    started = time.perf_counter()
    chunks = load_and_split()
    inventory = prepare(chunks)
    print(f"{args.dataset.name}: {len(cases)} cases")
    documents = {chunk.doc_id for chunk in inventory.chunks}
    print(f"langchain: {len(chunks):,} chunks from {len(documents)} PDFs")

    resolved = resolve(cases, inventory)
    lost = [case.id for case in resolved if not case.expected.get("relevant")]
    print(f"  labels no single chunk holds: {len(lost)}")

    run = run_eval(
        resolved,
        {"langchain": langchain_task(build_retriever(chunks, args.k))},
        graders(args.k),
        dataset=args.dataset.name,
        digest=digest(args.dataset),
        model=EMBED_MODEL,
        progress=False,
    )
    print(f"\n{len(run.rows)} rows in {time.perf_counter() - started:.1f}s, $0.00 spent\n")
    print(f"report: {write_report(run)}")


if __name__ == "__main__":
    main()
