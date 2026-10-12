"""Training data for the figure fine-tune, in two batches: Claude writes the questions a chunk of a
training report answers, then answers each with exactly the prompt the fine-tune is trained on.
`export` hands the answers and the task to underhood, which builds the conversations.

uv run python -m aigent.extraction.figure_training ask            # chunks, tokens, worst case
uv run python -m aigent.extraction.figure_training ask --yes      # send the question batch
uv run python -m aigent.extraction.figure_training answer --yes   # send the answer batch
uv run python -m aigent.extraction.figure_training export --out ../underhood/data/figures-v3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, JsonValue

from aigent.config import (
    CLIENT,
    FIGURE_ANSWER_CACHE,
    FIGURE_ANSWER_SLICE,
    FIGURE_LABEL_CORE_PER_DOC,
    FIGURE_LABEL_MODEL,
    FIGURE_LABEL_PLAIN_PER_DOC,
    FIGURE_LABEL_QUESTIONS,
    FIGURE_LABEL_SCALED_PER_DOC,
    FIGURE_LABEL_SEED,
    MAX_USD_FIGURE_LABELS,
    THINKING_EVAL_PARAM,
    max_tokens,
)
from aigent.extraction.figures import (
    CHANGE,
    METRICS,
    RATE_METRICS,
    SYSTEM,
    TOOL,
    Labelled,
    figure_messages,
)
from aigent.llm.adapters import spec
from aigent.llm.core import Llm, Request
from aigent.llm.messages import Failed, Parsed
from aigent.llm.pricing import Budget, estimate_eval_usd
from aigent.retrieval.chunk import Chunk, Inventory, by_sentence
from aigent.retrieval.corpus import TRAINING_DIR, TRAINING_MANIFEST, load_corpus

QUESTIONS = TRAINING_DIR / "figure-questions.jsonl"
ANSWERS = TRAINING_DIR / "figure-answers.jsonl"
ADJUDICATIONS = (
    Path(__file__).resolve().parents[3] / "evals" / "datasets" / "figure-adjudications.jsonl"
)
_CORE = re.compile(
    r"\b(revenue|turnover|ebitda|profit after tax|net profit|earnings per share|eps|dividend|"
    r"margin|borrowings|net debt|capital expenditure|capex|cash flow)\b",
    re.I,
)
_SCALED = re.compile(r"\d[\d,.]*\s*(%|per cent|lakh|crore|million|billion)", re.I)
_PLAIN = re.compile(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d{3,}\b")
_YEAR = re.compile(r"(19|20)\d\d")


class Questions(BaseModel):
    """What the question writer returns for one passage."""

    questions: list[str] = Field(
        description=(
            f"Up to {FIGURE_LABEL_QUESTIONS} questions, each naming the company and answered by "
            "figures the passage states; empty when it states none worth asking about."
        )
    )


class Taught(BaseModel):
    """A `Labelled` figure as the teacher's schema: the rules are in its system prompt already, so
    only the call, which that prompt does not ask for, carries one."""

    metric: str
    value: float
    unit: str | None
    year: int | None
    description: str | None
    call: str | None = Field(description=Labelled.model_fields["call"].description)


class Answer(BaseModel):
    """What the teacher returns for one question: the figures, each with its calculate call."""

    figures: list[Taught] = Field(
        description=(
            "Every figure in the passage that answers what is wanted, in the order it appears; "
            "empty when none does."
        )
    )


QUESTION_SYSTEM = (
    "You write questions for training a model that extracts figures from passages of Indian "
    "companies' annual reports. For the passage, write up to "
    f"{FIGURE_LABEL_QUESTIONS} questions an analyst might ask that it answers with figures. Each "
    "names the company and asks for one kind of figure, which may be a figure for two years. "
    "Prefer what a financial analyst looks for: "
    + ", ".join(m for m in METRICS if m != "other")
    + ". A question the passage cannot answer with a figure is wrong."
)


def source_of(doc_id: str, manifest: Path = TRAINING_MANIFEST) -> str:
    """How a report names itself in the prompt: company, then title."""
    documents = cast(dict[str, dict[str, str]], json.loads(manifest.read_text())["documents"])
    entry = documents[doc_id]
    return f"{entry['company']}, {entry['title']}"


def _kind(chunk: Chunk) -> str | None:
    states = any(not _YEAR.fullmatch(n.replace(",", "")) for n in _PLAIN.findall(chunk.text))
    if _CORE.search(chunk.text) and (states or _SCALED.search(chunk.text)):
        return "core"
    if _SCALED.search(chunk.text):
        return "scaled"
    return "plain" if states else None


def select(chunks: Sequence[Chunk], seed: int = FIGURE_LABEL_SEED) -> list[Chunk]:
    """Per report, a seeded sample of chunks naming a core metric beside a figure, then of chunks
    stating a scaled figure, then a plain one, in id order. Only years state nothing."""
    rng = random.Random(seed)
    wanted = {
        "core": FIGURE_LABEL_CORE_PER_DOC,
        "scaled": FIGURE_LABEL_SCALED_PER_DOC,
        "plain": FIGURE_LABEL_PLAIN_PER_DOC,
    }
    picked: list[Chunk] = []
    for doc_id in sorted({c.doc_id for c in chunks}):
        for kind, n in wanted.items():
            pool = [c for c in chunks if c.doc_id == doc_id and _kind(c) == kind]
            picked += rng.sample(pool, min(n, len(pool)))
    return sorted(picked, key=lambda c: c.id)


def ask_request(chunk: Chunk, source: str, cap: int) -> Request:
    """The question writer's call for one chunk."""
    user = f"<source>{source}</source>\n<passage>{chunk.text}</passage>"
    return Request(
        f"ask:{chunk.id}",
        [{"role": "user", "content": user}],
        cap,
        system=QUESTION_SYSTEM,
        model=FIGURE_LABEL_MODEL,
        thinking=THINKING_EVAL_PARAM,
    )


def answer_request(passage: str, source: str, question: str, step: str, cap: int) -> Request:
    """The teacher's call: the fine-tune's own prompt, on the labelling model, its system prompt
    marked for the cache every answer shares."""
    system, user = figure_messages(passage, source, question)
    return Request(
        step,
        [{"role": "user", "content": user}],
        cap,
        system=[{"type": "text", "text": system, "cache_control": FIGURE_ANSWER_CACHE}],
        model=FIGURE_LABEL_MODEL,
        thinking=THINKING_EVAL_PARAM,
    )


def _chunks() -> list[Chunk]:
    documents = load_corpus(TRAINING_DIR, cache=True)
    return list(Inventory.build("training", documents, by_sentence()).chunks)


def _llm(client: str) -> Llm:
    return Llm(client, budget=Budget(limit_usd=MAX_USD_FIGURE_LABELS, scope="figure labels"))


def _price(llm: Llm, requests: Sequence[Request], schema: type[BaseModel], cap: int) -> float:
    """Print a batch's size and worst case, counted for free off its longest request."""
    longest = max(requests, key=lambda r: len(str(r.messages[0]["content"])))
    tokens = llm.count(longest, schema)
    worst = estimate_eval_usd(FIGURE_LABEL_MODEL, len(requests), tokens, cap, batched=True)
    print(f"{len(requests)} requests on {FIGURE_LABEL_MODEL}")
    print(f"longest request {tokens} input tokens; output capped at {cap}")
    print(f"worst case ${worst:.2f} at the batch price; ceiling ${MAX_USD_FIGURE_LABELS:.2f}")
    return worst


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def ask(args: argparse.Namespace) -> None:
    """The question batch: price it, and with `--yes` send it and write the questions."""
    chosen = select(_chunks())
    if args.sample is not None:
        chosen = chosen[:: max(1, len(chosen) // args.sample)][: args.sample]
    llm = _llm(args.client)
    cap = max_tokens("FIGURE_QUESTIONS", spec(llm.client))
    sources = {doc_id: source_of(doc_id) for doc_id in {c.doc_id for c in chosen}}
    requests = [ask_request(c, sources[c.doc_id], cap) for c in chosen]
    print(f"{len(chosen)} chunks from {len(sources)} reports")
    _price(llm, requests, Questions, cap)
    if not args.yes:
        print("\nnothing spent. re-run with --yes to send the batch.")
        return
    results = llm.batch_records(requests, Questions)
    with QUESTIONS.open("w", encoding="utf-8") as out:
        for chunk, result in zip(chosen, results, strict=True):
            if isinstance(result, Failed) or result.parsed is None:
                continue
            row = {
                "chunk_id": chunk.id,
                "source": sources[chunk.doc_id],
                "passage": chunk.text,
                "questions": result.parsed.questions,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"questions: {QUESTIONS}; spent ${llm.spent_usd:.4f}")


def _key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["chunk_id"]), str(row["question"])


def _merged(
    old: Sequence[dict[str, object]], fresh: dict[tuple[str, str], dict[str, object]]
) -> list[dict[str, object]]:
    """The old answers in their order, each replaced by its fresh one where there is one."""
    return [fresh.get(_key(row), row) for row in old]


def answer(args: argparse.Namespace) -> None:
    """One request per question: price them, and with `--yes` send the first as an ordinary call,
    which writes the cached prompt, then the rest as a batch in slices that can read it. With
    `--only`, just the (chunk_id, question) pairs it lists, merged into the answers already there;
    a listed question whose relabel fails keeps its old answer."""
    only = None if args.only is None else {_key(r) for r in _rows(args.only)}
    asked = [
        (row, str(question))
        for row in _rows(QUESTIONS)
        for question in cast(list[str], row["questions"])
        if only is None or (str(row["chunk_id"]), str(question)) in only
    ]
    llm = _llm(args.client)
    cap = max_tokens("FIGURE_ANSWER", spec(llm.client))
    requests = [
        answer_request(str(row["passage"]), str(row["source"]), q, f"answer:{row['chunk_id']}", cap)
        for row, q in asked
    ]
    _price(llm, requests, Answer, cap)
    if not args.yes:
        print("\nnothing spent. re-run with --yes to send the batch.")
        return
    results: list[Parsed[Answer] | Failed] = [llm.parse(requests[0], Answer)]
    for start in range(1, len(requests), FIGURE_ANSWER_SLICE):
        results += llm.batch_records(requests[start : start + FIGURE_ANSWER_SLICE], Answer)
    batched = [r for r in results[1:] if isinstance(r, Parsed)]
    hits = sum(1 for r in batched if r.usage.cache_read_tokens)
    print(f"cache read by {hits} of {len(batched)} batched answers")
    fresh: dict[tuple[str, str], dict[str, object]] = {}
    for (row, question), result in zip(asked, results, strict=True):
        if isinstance(result, Failed) or result.parsed is None:
            continue
        kept = {k: row[k] for k in ("chunk_id", "source", "passage")}
        figures = result.parsed.model_dump()["figures"]
        fresh[(str(row["chunk_id"]), question)] = kept | {"question": question, "figures": figures}
    rows = list(fresh.values()) if only is None else _merged(_rows(ANSWERS), fresh)
    with ANSWERS.open("w", encoding="utf-8") as out:
        for answered in rows:
            out.write(json.dumps(answered) + "\n")
    print(f"answers: {ANSWERS}; spent ${llm.spent_usd:.4f}")


def task() -> dict[str, JsonValue]:
    """The task as underhood builds from it: the system prompt, the user turn as a template, the
    tool, the metric vocabulary, the metrics a percentage may be, and the word for a change."""
    _, user = figure_messages("{passage}", "{source}", "{wanted}")
    return {
        "system": SYSTEM,
        "user": user,
        "tool": TOOL,
        "metrics": list(METRICS),
        "rate_metrics": list(RATE_METRICS),
        "change": CHANGE,
    }


def export(answers: Path, out_dir: Path, adjudications: Path = ADJUDICATIONS) -> tuple[int, int]:
    """The labelled answers, with each hand-written `answer` in `adjudications` swapped in for its
    (chunk_id, question), and the task into `out_dir` for `underhood-tool-data`; how many answers
    went and how many were overridden."""
    lines = adjudications.read_text(encoding="utf-8").splitlines()
    rulings = [json.loads(x) for x in lines if x.startswith("{")]
    written = {_key(r): r["answer"] for r in rulings if "answer" in r}
    rows: list[dict[str, object]] = []
    overridden = 0
    for row in _rows(answers):
        if _key(row) in written:
            row = row | {"figures": written[_key(row)]}
            overridden += 1
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answers.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (out_dir / "task.json").write_text(json.dumps(task(), indent=2, ensure_ascii=False) + "\n")
    return len(rows), overridden


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aigent.extraction.figure_training")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, what in (("ask", "write questions for training chunks"), ("answer", "answer them")):
        command = commands.add_parser(name, help=f"{what} through the Batch API")
        command.add_argument("--yes", action="store_true", help="send the batch; else estimate")
        command.add_argument("--client", default=CLIENT, help=f"who is billed (default {CLIENT})")
        if name == "ask":
            command.add_argument("--sample", type=int, metavar="N", help="N chunks, spread out")
        else:
            command.add_argument(
                "--only", type=Path, default=None, help="relabel just these (underhood's dropped)"
            )
    exporting = commands.add_parser("export", help="write the answers and the task for underhood")
    exporting.add_argument("--out", type=Path, required=True, help="underhood's data directory")
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.command == "ask":
        ask(args)
    elif args.command == "answer":
        answer(args)
    else:
        n, overridden = export(ANSWERS, args.out)
        print(f"{n} answers ({overridden} overridden by hand) and the task -> {args.out}")


if __name__ == "__main__":
    main()
