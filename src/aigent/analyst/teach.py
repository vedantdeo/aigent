"""v4's training trajectories: the teacher runs the report analyst's own loop over the held-out
training reports, on the questions written for v3, writing each figure's calculate call for code
to make, as v3's teacher did; a trajectory is kept when it passes the specialist's check, its
descriptions keep their rule, and it finds every figure labelled for its question. `export` hands
the kept ones to underhood. Estimates and exits unless `--yes`; a run skips the questions its file
already holds.

uv run python -m aigent.analyst.teach run --sample 20
uv run python -m aigent.analyst.teach run --yes
uv run python -m aigent.analyst.teach export --out ../underhood/data/research-v4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, JsonValue, ValidationError

from aigent.analyst.cfg_analyst import SPECIALIST_TOOL_TURNS, max_tokens
from aigent.analyst.run import in_parallel
from aigent.analyst.served import opening, research, turn_request
from aigent.analyst.specialists import FigureJob, Found, Invalid, Told, check_figures
from aigent.analyst.toolbox import SEARCH
from aigent.config import (
    CLIENT,
    FIGURE_DESCRIPTION_WORDS,
    MAX_USD_RESEARCH_TEACH,
    RESEARCH_TEACH_CACHE,
    RESEARCH_TEACH_NAMED_SHARE,
    RESEARCH_TEACH_PAIRED_SHARE,
    RESEARCH_TEACH_QUESTIONS,
    RESEARCH_TEACH_SEED,
    RESEARCH_TEACH_WORKERS,
    RESEARCH_TEACHER_MODEL,
    RESEARCH_TYPICAL_OUTPUT,
    RESEARCH_TYPICAL_TURNS,
    SEARCH_METHOD,
    SEARCH_RESULT_TOKENS,
)
from aigent.extraction.figures import TOOL as CALCULATE_FUNCTION
from aigent.extraction.figures import Figure, Labelled, described, same_figure
from aigent.extraction.research import REPORT_SEARCH_TOOL, SourcedFigure, research_messages
from aigent.llm.adapters import spec
from aigent.llm.adapters.cfg_hostllm import ANALYST
from aigent.llm.adapters.openai import OpenAI, function_tool
from aigent.llm.core import Dispatch, Llm, Request
from aigent.llm.messages import Block, Msg, Tool
from aigent.llm.pricing import Budget, cost_usd
from aigent.retrieval.chunk import Chunk, Inventory, by_sentence, context_block
from aigent.retrieval.corpus import TRAINING_DIR, TRAINING_MANIFEST, load_corpus
from aigent.retrieval.rank import Indexes, build_ranker
from aigent.tools import CALCULATOR_TOOL, execute_tool, fence

ANSWERS = TRAINING_DIR / "figure-answers.jsonl"
TRAJECTORIES = TRAINING_DIR / "research-trajectories.jsonl"
SCALE_POWERS = (3, 5, 6, 7, 9)  # thousand, lakh, million, crore, billion
# What the teacher is told beyond the trained prompt: it writes each call, and code makes it.
TEACHER_NOTE = (
    "You are writing a worked example of this job and have no calculate tool here. Give each "
    "figure a call instead: the calculate expression behind its value, the number as printed "
    "times its scale or over 100, '1,250 crore' is '1250 * 10**7' and '7.25%' is '7.25 / 100', "
    "or null when the number is printed in full. Never write a scaled amount or a fraction "
    "without its call."
)
TEACHER_TOOLS: tuple[Tool, ...] = (REPORT_SEARCH_TOOL,)

# The metrics added on 2026-10-19 and the v3 `other` descriptions each reads back, METRICS order.
RELABEL: dict[str, re.Pattern[str]] = {
    "tax": re.compile(r"tax expense|current tax|effective (income )?tax rate", re.I),
    "receivables": re.compile(r"receivable", re.I),
    "equity": re.compile(
        r"total equity|other equity|shareholders'? (equity|funds)|net worth|share capital|^equity$",
        re.I,
    ),
    "assets": re.compile(r"\bassets\b", re.I),
}


@dataclass(frozen=True)
class Asked:
    """A v3 question, the chunk it was written for, and the figures labelled as its answer;
    `paired` when the question also asks about another company, whose part is not this job's."""

    chunk_id: str
    question: str
    labels: tuple[Labelled, ...]
    paired: bool = False

    @property
    def report(self) -> str:
        """The report it is about: BAJAJ-AUTO-FY24 for BAJAJ-AUTO-FY24#0004."""
        return self.chunk_id.split("#")[0]

    @property
    def job(self) -> FigureJob:
        ticker, _, year = self.report.rpartition("-FY")
        return FigureJob(self.question, ticker, 2000 + int(year))


def companies(manifest: Path = TRAINING_MANIFEST) -> dict[str, str]:
    """Each training report's ticker and the company it names."""
    documents = cast(dict[str, dict[str, str]], json.loads(manifest.read_text())["documents"])
    return {entry["ticker"]: entry["company"] for entry in documents.values()}


def relabelled(label: Labelled) -> Labelled:
    """A v3 label in the current vocabulary: an `other` whose description names an added metric
    becomes that metric, money only, but for the tax rate."""
    if label.metric != "other" or not label.description:
        return label
    for metric, named in RELABEL.items():
        rate = metric == "tax" and "rate" in label.description.lower()
        if named.search(label.description) and (label.unit is not None or rate):
            return label.model_copy(update={"metric": metric, "description": None})
    return label


def load_asked(answers: Path = ANSWERS) -> list[Asked]:
    """Every labelled question with at least one figure, its labels in the current vocabulary,
    in file order."""
    found: list[Asked] = []
    for line in answers.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        labels = tuple(relabelled(Labelled.model_validate(f)) for f in row["figures"])
        if labels:
            found.append(Asked(row["chunk_id"], row["question"], labels))
    return found


def named(asked: Asked) -> bool:
    """Whether a question asks for a named metric: any of its labels is not `other`."""
    return any(label.metric != "other" for label in asked.labels)


def _evenly(asked: Sequence[Asked], n: int, rng: random.Random) -> list[Asked]:
    """`n` of `asked`, round-robin across their reports, each report's in a seeded order."""
    by_report: dict[str, list[Asked]] = {}
    for one in asked:
        by_report.setdefault(one.report, []).append(one)
    queues = [rng.sample(by_report[r], len(by_report[r])) for r in sorted(by_report)]
    picked: list[Asked] = []
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop(0))
    return picked


def _split(asked: Sequence[Asked], n: int, seed: int, share: float) -> list[Asked]:
    """`n` single questions, `share` of them named and the rest `other`, a part too small for its
    share made up by the other, each drawn evenly across the reports in a seeded order."""
    names = [a for a in asked if named(a)]
    others = [a for a in asked if not named(a)]
    rest = min(n - round(n * share), len(others))
    first = min(n - rest, len(names))
    rest = min(n - first, len(others))
    return [
        *_evenly(names, first, random.Random(seed)),
        *_evenly(others, rest, random.Random(seed + 1)),
    ]


def _paired(asked: Sequence[Asked], n: int, rng: random.Random) -> list[Asked]:
    """`n` questions that each join one named question to another about a different company, in
    a random order; the job, its report and its labels are the first's. None is used twice."""
    ordered = _evenly(asked, len(asked), rng)
    taken: set[tuple[str, str]] = set()
    made: list[Asked] = []
    for job in ordered:
        if len(made) == n:
            break
        if (job.chunk_id, job.question) in taken:
            continue
        partner = next(
            (
                other
                for other in ordered
                if (other.chunk_id, other.question) not in taken
                and other.job.ticker != job.job.ticker
            ),
            None,
        )
        if partner is None:
            continue
        taken |= {(job.chunk_id, job.question), (partner.chunk_id, partner.question)}
        parts = [job.question, partner.question]
        if rng.random() < 0.5:
            parts.reverse()
        made.append(Asked(job.chunk_id, " ".join(parts), job.labels, paired=True))
    return made


def select(
    asked: Sequence[Asked],
    n: int,
    seed: int = RESEARCH_TEACH_SEED,
    share: float = RESEARCH_TEACH_NAMED_SHARE,
    paired: float = RESEARCH_TEACH_PAIRED_SHARE,
) -> list[Asked]:
    """`n` questions: `paired` of them asking about two companies, the job naming one, and the
    rest single, `share` of those named and the rest `other`; all drawn evenly across the
    reports in a seeded order, no question in two of them."""
    singles = _split(asked, n - round(n * paired), seed, share)
    used = {(a.chunk_id, a.question) for a in singles}
    pool = [a for a in asked if named(a) and (a.chunk_id, a.question) not in used]
    pairs = _paired(pool, n - len(singles), random.Random(seed + 2))
    return [*singles, *pairs]


def mix(asked: Sequence[Asked]) -> dict[str, int]:
    """What a draw holds: single named and other questions, paired ones, how many ask for more
    than one year, and how many of those are paired too."""
    singles = [a for a in asked if not a.paired]
    years = [a for a in asked if len({f.year for f in a.labels if f.year}) > 1]
    return {
        "named": sum(named(a) for a in singles),
        "other": sum(not named(a) for a in singles),
        "two companies": len(asked) - len(singles),
        "two years or more": len(years),
        "both": sum(a.paired for a in years),
    }


def report_search(find: Callable[[str, str], list[Chunk]], reports: Collection[str]) -> Dispatch:
    """`search_filings` over the training reports, one report a call, by ticker and year: a
    report not in `reports` is an error, never an exception."""

    def dispatch(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        if name != SEARCH:
            return f"Error: {name} is not available here", True
        year = str(arguments.get("year", ""))
        if not year.isdigit():
            return f"Error: no financial year {year!r}", True
        doc_id = f"{arguments.get('company')}-FY{int(year) % 100:02d}"
        if doc_id not in reports:
            return f"Error: no training report {doc_id}", True
        found = find(str(arguments.get("query", "")), doc_id)
        return fence(SEARCH, context_block(found) or "No passages matched that query."), False

    return dispatch


def training_search(cache: bool = True) -> Dispatch:
    """The training reports chunked, embedded and indexed as the filings are, searched alike."""
    documents = load_corpus(TRAINING_DIR, cache=cache)
    inventory = Inventory.build("by_sentence", documents, by_sentence())
    ranker = build_ranker(SEARCH_METHOD, Indexes.build(inventory, cache=cache))
    one_at_a_time = threading.Lock()  # torch on MPS is not safe across threads

    def find(query: str, doc_id: str) -> list[Chunk]:
        with one_at_a_time:
            hits = ranker.rank(query, doc_id=frozenset({doc_id}))
        return [inventory.by_id[hit.chunk_id] for hit in hits]

    return report_search(find, {document.doc_id for document in documents})


def found_as(label: Labelled, figure: Figure) -> bool:
    """Whether `figure` is `label`: the same metric, value, unit and year, or, for a label written
    as printed, the value a named scale gives it, which the label's chunk may not have shown."""
    if same_figure(label, figure):
        return True
    scaled = (label.model_copy(update={"value": label.value * 10**k}) for k in SCALE_POWERS)
    return label.call is None and any(same_figure(s, figure) for s in scaled)


def missed(labels: Sequence[Labelled], figures: Sequence[Figure]) -> list[Labelled]:
    """The labelled figures no answered figure is."""
    return [label for label in labels if not any(found_as(label, f) for f in figures)]


class TaughtFigure(SourcedFigure):
    """A teacher's figure: with the calculate call behind its value, which code then makes."""

    call: str | None = Field(description=Labelled.model_fields["call"].description)


class Taught(BaseModel):
    figures: list[TaughtFigure]


def _taught(text: str) -> Taught | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return Taught.model_validate(json.loads(text[start : end + 1]))
    except (ValueError, ValidationError):
        return None


def worked(told: Told) -> Told:
    """The teacher's trajectory as the specialist runs it: its searches, then every figure's call
    made as one calculate turn, then the answer without the calls. Invalid for a reply holding no
    figures, a call that does not give its figure's value, or more turns than the loop allows."""
    taught = _taught(told.text)
    if taught is None:
        raise Invalid("the reply holds no figures")
    called = [f for f in taught.figures if f.call]
    turns: list[Msg] = []
    if called:
        uses: list[Block] = []
        results: list[Block] = []
        for n, figure in enumerate(called):
            expression = str(figure.call)
            content, is_error = execute_tool(CALCULATOR_TOOL["name"], {"expression": expression})
            try:
                gives = float(content)
            except ValueError:
                is_error, gives = True, math.nan
            if is_error or not math.isclose(gives, figure.value, rel_tol=1e-9, abs_tol=1e-12):
                raise Invalid(f"{expression} gives {content}, not {figure.value}")
            use = {"type": "tool_use", "id": f"calc_{n}", "name": "calculate"}
            uses.append({**use, "input": {"expression": expression}})
            results.append({"type": "tool_result", "tool_use_id": f"calc_{n}", "content": content})
        turns = [{"role": "assistant", "content": uses}, {"role": "user", "content": results}]
    answer = json.dumps({"figures": [f.model_dump(exclude={"call"}) for f in taught.figures]})
    final: Msg = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    conversation = (*told.conversation[:-1], *turns, final)
    if sum(m["role"] == "assistant" for m in conversation) > SPECIALIST_TOOL_TURNS:
        raise Invalid(f"more than the loop's {SPECIALIST_TOOL_TURNS} turns")
    calls = tuple(str(f.call) for f in called)
    return replace(told, text=answer, calls=calls, conversation=conversation)


def undescribed(found: Found) -> list[str]:
    """The figures whose description breaks its rule: verbatim from the passage and at most
    `FIGURE_DESCRIPTION_WORDS` words for an other figure, none or 'change' for a named one."""
    bad: list[str] = []
    for f in found.figures:
        long = f.metric == "other" and len((f.description or "").split()) > FIGURE_DESCRIPTION_WORDS
        if long or not described(f, found.passages[f.chunk_id].text):
            bad.append(f"{f.metric} {f.value:g} described {f.description!r}")
    return bad


def teach(asked: Asked, run: Callable[[FigureJob], Told]) -> dict[str, JsonValue]:
    """One trajectory as a row: the question, every turn as the specialist would take them, the
    figures, and whether it is kept — worked, passing the specialist's check, every description
    to its rule and every label found — or why not."""
    job = asked.job
    told = run(job)
    why: str | None = None
    figures: list[JsonValue] = []
    try:
        told = worked(told)
        found = check_figures(job, told)
    except Invalid as refused:
        why = str(refused)
    else:
        figures = [cast(JsonValue, f.model_dump()) for f in found.figures]
        if bad := undescribed(found):
            why = "; ".join(bad)
        elif lost := missed(asked.labels, found.figures):
            why = "missed " + "; ".join(f"{f.metric} {f.value:g} {f.year}" for f in lost)
    return {
        "chunk_id": asked.chunk_id,
        "question": asked.question,
        "ticker": job.ticker,
        "year": job.year,
        "paired": asked.paired,
        "kept": why is None,
        "why": why,
        "figures": figures,
        "usd": told.usd,
        "conversation": cast(JsonValue, json.loads(json.dumps(told.conversation, default=str))),
    }


def _key(row: dict[str, JsonValue]) -> tuple[str, str]:
    return str(row["chunk_id"]), str(row["question"])


def read_rows(path: Path) -> list[dict[str, JsonValue]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: Sequence[dict[str, JsonValue]]) -> None:
    """The rows sorted by chunk and question, one canonical order."""
    ordered = sorted(rows, key=_key)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))


def estimate(first_tokens: int, questions: int, model: str, cap: int) -> tuple[float, float]:
    """A typical and a worst run, in USD: typical is `RESEARCH_TYPICAL_TURNS` turns each adding a
    search result and writing `RESEARCH_TYPICAL_OUTPUT`; worst is every turn, every cap."""

    def run_of(turns: int, output: int) -> float:
        inputs = sum(first_tokens + n * (SEARCH_RESULT_TOKENS + output) for n in range(turns))
        return questions * cost_usd(model, inputs, turns * output)

    return run_of(RESEARCH_TYPICAL_TURNS, RESEARCH_TYPICAL_OUTPUT), run_of(
        SPECIALIST_TOOL_TURNS, cap
    )


def task() -> dict[str, JsonValue]:
    """The task as underhood builds from it: the system prompt and the two tools, as a Hugging
    Face chat template takes them."""
    system, _ = research_messages("", "", "")
    return {
        "system": system,
        "tools": [cast(JsonValue, function_tool(REPORT_SEARCH_TOOL)), CALCULATE_FUNCTION],
    }


def export(trajectories: Path, out_dir: Path) -> int:
    """Each kept trajectory into `out_dir` as the served model reads it, on the OpenAI wire with
    its system prompt first, and the task beside them; how many went."""
    wire = OpenAI(ANALYST)
    system = str(task()["system"])
    kept = [row for row in read_rows(trajectories) if row["kept"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as out:
        for row in sorted(kept, key=_key):
            turns = cast(list[Msg], row["conversation"])
            messages = wire.messages(Request("export", turns, 1, system=system))
            line = {"chunk_id": row["chunk_id"], "question": row["question"], "messages": messages}
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
    (out_dir / "task.json").write_text(json.dumps(task(), indent=2, ensure_ascii=False) + "\n")
    return len(kept)


def _run(args: argparse.Namespace) -> None:
    done = {_key(row) for row in read_rows(args.out)}
    chosen = select(load_asked(), args.sample or RESEARCH_TEACH_QUESTIONS)
    todo = [one for one in chosen if (one.chunk_id, one.question) not in done]
    names = companies()
    teacher = Llm(args.client, budget=Budget(limit_usd=MAX_USD_RESEARCH_TEACH, scope="teach"))
    cap = max_tokens("RESEARCH_TURN", spec(teacher.client))
    reports = len({one.report for one in chosen})
    print(f"{len(chosen)} questions over {reports} reports, {len(todo)} to run")
    print(", ".join(f"{count} {kind}" for kind, count in mix(chosen).items()))
    if not todo:
        return
    longest = max(todo, key=lambda one: len(one.question))
    system = f"{research_messages('', '', '')[0]}\n{TEACHER_NOTE}"
    first = turn_request(
        opening(longest.job, names), cap, RESEARCH_TEACHER_MODEL, None, system, TEACHER_TOOLS
    )
    tokens = teacher.count(first)
    typical, worst = estimate(tokens, len(todo), RESEARCH_TEACHER_MODEL, cap)
    print(f"teacher {RESEARCH_TEACHER_MODEL}; first turn {tokens} tokens, a free count")
    print(f"typical ${typical:.2f} uncached, worst ${worst:.2f}")
    print(f"ceiling ${MAX_USD_RESEARCH_TEACH:.2f}")
    if not args.yes:
        print("estimate only: nothing sent. Add --yes to run.")
        return
    search = training_search()
    loop = research(
        teacher, search, RESEARCH_TEACHER_MODEL, names, RESEARCH_TEACH_CACHE, system, TEACHER_TOOLS
    )
    finished = in_parallel(todo, lambda one: teach(one, loop), args.workers)
    rows = [row for row in finished if not isinstance(row, Exception)]
    for one, row in zip(todo, finished, strict=True):
        if isinstance(row, Exception):
            print(f"{one.chunk_id} failed: {type(row).__name__}: {row}")
    write_rows(args.out, [*read_rows(args.out), *rows])
    kept = sum(1 for row in rows if row["kept"])
    print(f"{kept} of {len(rows)} kept; spent ${teacher.spent_usd:.4f}; {args.out}")


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aigent.analyst.teach", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="run the teacher; without --yes, estimate")
    running.add_argument("--yes", action="store_true", help="send the calls")
    running.add_argument("--sample", type=int, metavar="N", help="the first N questions drawn")
    running.add_argument("--client", default=CLIENT, help=f"who is billed (default {CLIENT})")
    running.add_argument("--workers", type=int, default=RESEARCH_TEACH_WORKERS)
    running.add_argument("--out", type=Path, default=TRAJECTORIES)
    exporting = commands.add_parser("export", help="write the kept trajectories for underhood")
    exporting.add_argument("--out", type=Path, required=True, help="underhood's data directory")
    exporting.add_argument("--trajectories", type=Path, default=TRAJECTORIES)
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        _run(args)
    else:
        print(f"{export(args.trajectories, args.out)} trajectories to {args.out}")


if __name__ == "__main__":
    main()
