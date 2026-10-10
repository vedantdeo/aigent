"""Training data for the figure fine-tune: chunks of the training reports, labelled by Claude
through the Batch API, then written as tool-use conversations for underhood to train on.

uv run python -m aigent.extraction.figure_training label          # chunks, tokens, worst case
uv run python -m aigent.extraction.figure_training label --yes    # send the batch
uv run python -m aigent.extraction.figure_training build --out ../underhood/data/figures
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, ValidationError

from aigent.config import (
    CLIENT,
    FIGURE_LABEL_MODEL,
    FIGURE_LABEL_PLAIN_PER_DOC,
    FIGURE_LABEL_QUESTIONS,
    FIGURE_LABEL_SCALED_PER_DOC,
    FIGURE_LABEL_SEED,
    FIGURE_VALID_FRACTION,
    MAX_USD_FIGURE_LABELS,
    THINKING_EVAL_PARAM,
    max_tokens,
)
from aigent.extraction.figures import METRICS, TOOL, Labelled, Message, transcript
from aigent.llm.adapters import spec
from aigent.llm.core import Llm, Request
from aigent.llm.messages import Failed
from aigent.llm.pricing import Budget, estimate_eval_usd
from aigent.retrieval.chunk import Chunk, Inventory, by_sentence
from aigent.retrieval.corpus import TRAINING_DIR, TRAINING_MANIFEST, load_corpus

LABELS = TRAINING_DIR / "figure-labels.jsonl"
_SCALED = re.compile(r"\d[\d,.]*\s*(%|per cent|lakh|crore|million|billion)", re.I)
_PLAIN = re.compile(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d{3,}\b")
_YEAR = re.compile(r"(19|20)\d\d")


class Asked(BaseModel):
    """One question the passage answers, with the figures that answer it."""

    question: str = Field(
        description=(
            "A question an analyst might ask that the passage answers with figures, naming the "
            "company, as a reader of the report would phrase it."
        )
    )
    figures: list[Labelled] = Field(description="Every figure in the passage that answers it.")


class Labels(BaseModel):
    """What the labeller returns for one passage."""

    asked: list[Asked] = Field(
        description=(
            f"Up to {FIGURE_LABEL_QUESTIONS} questions, each answered by figures the passage "
            "states; empty when it states none worth asking about."
        )
    )


def _figure_rules() -> str:
    return "\n".join(f"- {name}: {f.description}" for name, f in Labelled.model_fields.items())


LABEL_SYSTEM = (
    "You write training labels for a model that extracts figures from passages of Indian "
    "companies' annual reports. For the passage, write up to "
    f"{FIGURE_LABEL_QUESTIONS} questions an analyst might ask that it answers with figures, "
    "and for each question every figure that answers it. Prefer the figures a financial "
    "analyst looks for: "
    + ", ".join(m for m in METRICS if m != "other")
    + ". Each figure:\n"
    + _figure_rules()
)


def source_of(doc_id: str, manifest: Path = TRAINING_MANIFEST) -> str:
    """How a report names itself in the prompt: company, then title."""
    documents = cast(dict[str, dict[str, str]], json.loads(manifest.read_text())["documents"])
    entry = documents[doc_id]
    return f"{entry['company']}, {entry['title']}"


def _kind(chunk: Chunk) -> str | None:
    if _SCALED.search(chunk.text):
        return "scaled"
    if any(not _YEAR.fullmatch(n.replace(",", "")) for n in _PLAIN.findall(chunk.text)):
        return "plain"
    return None


def select(chunks: Sequence[Chunk], seed: int = FIGURE_LABEL_SEED) -> list[Chunk]:
    """Per report, a seeded sample of chunks stating a scaled figure and of chunks stating a plain
    one, in id order. A chunk whose only numbers are years states neither."""
    rng = random.Random(seed)
    wanted = {"scaled": FIGURE_LABEL_SCALED_PER_DOC, "plain": FIGURE_LABEL_PLAIN_PER_DOC}
    picked: list[Chunk] = []
    for doc_id in sorted({c.doc_id for c in chunks}):
        for kind, n in wanted.items():
            pool = [c for c in chunks if c.doc_id == doc_id and _kind(c) == kind]
            picked += rng.sample(pool, min(n, len(pool)))
    return sorted(picked, key=lambda c: c.id)


def label_request(chunk: Chunk, source: str, cap: int) -> Request:
    """The labeller's call for one chunk, on the labelling model whatever the client serves."""
    user = f"<source>{source}</source>\n<passage>{chunk.text}</passage>"
    return Request(
        f"label:{chunk.id}",
        [{"role": "user", "content": user}],
        cap,
        system=LABEL_SYSTEM,
        model=FIGURE_LABEL_MODEL,
        thinking=THINKING_EVAL_PARAM,
    )


def _chunks() -> list[Chunk]:
    documents = load_corpus(TRAINING_DIR, cache=True)
    return list(Inventory.build("training", documents, by_sentence()).chunks)


def _llm(client: str) -> Llm:
    return Llm(client, budget=Budget(limit_usd=MAX_USD_FIGURE_LABELS, scope="figure labels"))


def label(args: argparse.Namespace) -> None:
    """Print the batch's size and worst case; with `--yes`, send it and write what came back."""
    chosen = select(_chunks())
    if args.sample is not None:
        chosen = chosen[:: max(1, len(chosen) // args.sample)][: args.sample]
    llm = _llm(args.client)
    settings = spec(llm.client)
    cap = max_tokens("FIGURE_LABELS", settings)
    sources = {doc_id: source_of(doc_id) for doc_id in {c.doc_id for c in chosen}}
    requests = [label_request(c, sources[c.doc_id], cap) for c in chosen]
    longest = max(requests, key=lambda r: len(str(r.messages[0]["content"])))
    tokens = llm.count(longest, Labels)
    worst = estimate_eval_usd(FIGURE_LABEL_MODEL, len(requests), tokens, cap, batched=True)
    print(f"{len(requests)} chunks from {len(sources)} reports on {FIGURE_LABEL_MODEL}")
    print(f"longest request {tokens} input tokens; output capped at {cap}")
    print(f"worst case ${worst:.2f} at the batch price; ceiling ${MAX_USD_FIGURE_LABELS:.2f}")
    if not args.yes:
        print("\nnothing spent. re-run with --yes to send the batch.")
        return
    results = llm.batch_records(requests, Labels)
    failed = 0
    with LABELS.open("w", encoding="utf-8") as out:
        for chunk, result in zip(chosen, results, strict=True):
            if isinstance(result, Failed) or result.parsed is None:
                failed += 1
                continue
            row = {
                "chunk_id": chunk.id,
                "source": sources[chunk.doc_id],
                "passage": chunk.text,
                "asked": result.parsed.model_dump()["asked"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"labels: {LABELS} ({len(results) - failed} chunks, {failed} without a record)")


@dataclass(frozen=True)
class Built:
    """What `build` wrote and what it dropped."""

    train: int
    valid: int
    dropped: int


def build(labels: Path, out_dir: Path, seed: int = FIGURE_LABEL_SEED) -> Built:
    """Every labelled question as a conversation, split by chunk so a passage never sits on both
    sides. A question with no figures, or one whose arithmetic does not check out, is dropped."""
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line]
    chunk_ids = sorted({str(row["chunk_id"]) for row in rows})
    held = set(random.Random(seed).sample(chunk_ids, round(len(chunk_ids) * FIGURE_VALID_FRACTION)))
    splits: dict[str, list[list[Message]]] = {"train": [], "valid": []}
    dropped = 0
    for row in rows:
        for asked in row["asked"]:
            messages: list[Message] | None = None
            try:
                figures = [Labelled.model_validate(f) for f in asked["figures"]]
                if figures:
                    messages = transcript(row["passage"], row["source"], asked["question"], figures)
            except (ValidationError, ValueError):
                messages = None
            if not messages:
                dropped += 1
                continue
            splits["valid" if row["chunk_id"] in held else "train"].append(messages)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, conversations in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for messages in conversations:
                out.write(json.dumps({"messages": messages, "tools": [TOOL]}) + "\n")
    return Built(len(splits["train"]), len(splits["valid"]), dropped)


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aigent.extraction.figure_training")
    commands = parser.add_subparsers(dest="command", required=True)
    labelling = commands.add_parser("label", help="label training chunks through the Batch API")
    labelling.add_argument("--yes", action="store_true", help="send the batch; else estimate")
    labelling.add_argument("--sample", type=int, metavar="N", help="label N chunks, spread out")
    labelling.add_argument("--client", default=CLIENT, help=f"who labels (default {CLIENT})")
    building = commands.add_parser("build", help="write the labels as training conversations")
    building.add_argument("--out", type=Path, required=True, help="directory for train/valid")
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.command == "label":
        label(args)
        return
    built = build(LABELS, args.out)
    print(f"{built.train} train, {built.valid} valid, {built.dropped} dropped -> {args.out}")


if __name__ == "__main__":
    main()
