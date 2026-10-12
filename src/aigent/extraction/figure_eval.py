"""The figure fine-tune's eval: the hand-labelled test set as prompts a model answers elsewhere, and
the grade of what it answered, figures and tool use alike.

uv run python -m aigent.extraction.figure_eval export --out ../underhood/data/figures/test.jsonl
uv run python -m aigent.extraction.figure_eval grade ../underhood/data/figures/eval-tuned.jsonl
uv run python -m aigent.extraction.figure_eval rows \
    --conversations ../underhood/data/figures-v2/valid.jsonl \
    --out ../underhood/data/figures-v2/census.jsonl
uv run python -m aigent.extraction.figure_eval census predictions.jsonl --test census.jsonl

A prediction is one JSON line: {"id", "final" (the last reply), "calls" (each calculate expression
the model made), "error" (or null)}.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from aigent.config import FIGURE_LABEL_SEED
from aigent.evals.dataset import Case, load_jsonl
from aigent.evals.grade import Outcome
from aigent.extraction.figure_training import source_of
from aigent.extraction.figures import (
    CURRENCIES,
    DATASET,
    TOOL,
    Figure,
    Figures,
    _described,
    _printed,
    _same,
    figure_messages,
    figures_match,
    parse_figures,
    unit_key,
)
from aigent.retrieval.chunk import Inventory, by_sentence
from aigent.retrieval.corpus import MANIFEST, load_corpus


def passages() -> dict[str, str]:
    """Every test-corpus chunk's text by id, under the strategy the test set's ids came from."""
    inventory = Inventory.build("by_sentence", load_corpus(cache=True), by_sentence())
    return {chunk.id: chunk.text for chunk in inventory.chunks}


def export(cases: Sequence[Case], texts: dict[str, str]) -> list[dict[str, JsonValue]]:
    """Each case as the prompt the fine-tune is trained on, with its passage, source and label."""
    rows: list[dict[str, JsonValue]] = []
    for case in cases:
        chunk_id = str(case.input["chunk_id"])
        passage = texts[chunk_id]
        source = source_of(chunk_id.split("#")[0], MANIFEST)
        system, user = figure_messages(passage, source, str(case.input["wanted"]))
        rows.append(
            {
                "id": case.id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "tools": [TOOL],
                "passage": passage,
                "expected": case.expected["figures"],
            }
        )
    return rows


_PASSAGE = re.compile(r"<passage>(.*)</passage>", re.S)
_WANTED = re.compile(r"<wanted>(.*)</wanted>", re.S)
_EXPRESSION = re.compile(r"\d\s*\*\*\s*\d")


def _label_calls(messages: Sequence[dict[str, JsonValue]]) -> list[str]:
    """The calculate expressions a training conversation's label made."""
    out: list[str] = []
    for message in messages:
        for call in cast(
            list[dict[str, dict[str, dict[str, str]]]], message.get("tool_calls") or []
        ):
            out.append(call["function"]["arguments"]["expression"])
    return out


def from_conversations(
    conversations: Sequence[dict[str, JsonValue]],
    prefix: str = "va",
    originals: set[tuple[str, str]] | None = None,
) -> list[dict[str, JsonValue]]:
    """Training-format conversations as eval rows: the prompt, the passage, the final figures as
    the label and the calls behind them, ids `<prefix>-0000` on by position. With `originals`,
    the (passage, question) pairs labelled by hand, each row also says whether it is a copy."""
    rows: list[dict[str, JsonValue]] = []
    for i, conversation in enumerate(conversations):
        messages = cast(list[dict[str, JsonValue]], conversation["messages"])
        user = str(messages[1]["content"])
        passage, wanted = _PASSAGE.search(user), _WANTED.search(user)
        row: dict[str, JsonValue] = {
            "id": f"{prefix}-{i:04d}",
            "messages": cast(JsonValue, messages[:2]),
            "tools": conversation["tools"],
            "passage": passage.group(1) if passage else "",
            "expected": json.loads(str(messages[-1]["content"]))["figures"],
            "calls": cast(JsonValue, _label_calls(messages)),
        }
        if originals is not None:
            asked = (str(row["passage"]), wanted.group(1) if wanted else "")
            row["augmented"] = asked not in originals
        rows.append(row)
    return rows


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _scaled(want: float, got: float) -> bool:
    """Whether `got` is `want` off by a power of ten other than one."""
    if want == 0 or got == 0:
        return False
    power = math.log10(abs(got / want))
    return abs(power - round(power)) < 1e-6


def _differences(want: Figure, got: Figure, passage: str) -> list[str]:
    """What separates two figures with the same value."""
    out: list[str] = []
    if want.metric != got.metric:
        out.append("metric")
    if unit_key(want.unit) != unit_key(got.unit):
        out.append(
            "unit not a currency" if unit_key(got.unit) not in (*CURRENCIES, None) else "unit"
        )
    if want.year != got.year:
        out.append(
            "year added"
            if want.year is None
            else "year dropped"
            if got.year is None
            else "year wrong"
        )
    if not _described(got, passage):
        out.append("description")
    return out


def faults(case_row: dict[str, JsonValue], prediction: dict[str, JsonValue]) -> list[str]:
    """Every way a prediction departs from its case, by kind: empty for a perfect answer."""
    final = str(prediction.get("final") or "")
    passage = str(case_row["passage"])
    called = bool(prediction.get("calls"))
    out: list[str] = []
    if needs_call(case_row) and not called:
        out.append("no call when needed")
    if not needs_call(case_row) and called:
        out.append("call when not needed")
    got_figures = parse_figures(final)
    if got_figures is None:
        out.append("expression in JSON" if _EXPRESSION.search(final) else "invalid JSON")
        return out
    want = Figures.model_validate({"figures": case_row["expected"]}).figures
    left = list(got_figures.figures)
    for figure in want:
        exact = next((g for g in left if _same(figure, g) and _described(g, passage)), None)
        if exact is not None:
            left.remove(exact)
            continue
        same = next((g for g in left if _close(g.value, figure.value)), None)
        raw = next((g for g in left if _close(g.value, figure.value * 100)), None)
        scaled = next((g for g in left if _scaled(figure.value, g.value)), None)
        if same is not None:
            out += _differences(figure, same, passage)
            left.remove(same)
        elif raw is not None:
            out.append("percentage not converted")
            left.remove(raw)
        elif scaled is not None:
            out.append("wrong scale")
            left.remove(scaled)
        else:
            out.append("missed")
    out += ["extra"] * len(left)
    return out


def census(
    rows: Sequence[dict[str, JsonValue]], predictions: dict[str, dict[str, JsonValue]]
) -> tuple[Counter[str], dict[str, list[str]]]:
    """How often each fault occurs across the rows, and each row's faults; a missing prediction is
    one `no prediction`."""
    counts: Counter[str] = Counter()
    each: dict[str, list[str]] = {}
    for row in rows:
        prediction = predictions.get(str(row["id"]))
        found = ["no prediction"] if prediction is None else faults(row, prediction)
        each[str(row["id"])] = found
        counts.update(found)
    return counts, each


def needs_call(case_row: dict[str, JsonValue]) -> bool:
    """Whether answering a case takes a calculate call: the label made one, where the row carries
    its calls; else some expected value is not printed as is."""
    if "calls" in case_row:
        return bool(case_row["calls"])
    figures = Figures.model_validate({"figures": case_row["expected"]}).figures
    printed = _printed(str(case_row["passage"]))
    return any(abs(f.value) not in printed for f in figures)


@dataclass(frozen=True)
class Graded:
    """One arm's numbers over the test set."""

    cases: int
    passed: int
    parsed: int
    recall: float
    all_found: int
    no_extra: int
    described: int
    called_when_needed: int
    needed: int
    called_when_not: int
    not_needed: int

    def lines(self) -> list[str]:
        """The arm as the report prints it."""
        n = self.cases
        return [
            f"passed {self.passed}/{n}  (all found {self.all_found}, nothing extra "
            f"{self.no_extra}, described {self.described})",
            f"valid JSON {self.parsed}/{n}; mean recall {self.recall:.2f}",
            f"called calculate when needed {self.called_when_needed}/{self.needed}; "
            f"when not needed {self.called_when_not}/{self.not_needed}",
        ]


def grade(
    rows: Sequence[dict[str, JsonValue]], predictions: dict[str, dict[str, JsonValue]]
) -> Graded:
    """Score each exported case against its prediction; a missing one counts as a failed case."""
    passed = parsed = all_found = no_extra = described = 0
    recall = 0.0
    hit = needed = false_call = not_needed = 0
    for row in rows:
        prediction = predictions.get(str(row["id"]), {})
        figures = parse_figures(str(prediction.get("final") or ""))
        output: dict[str, JsonValue] = {}
        if figures is not None:
            parsed += 1
            output = {"figures": cast(JsonValue, figures.model_dump()["figures"])}
        case = Case(
            id=str(row["id"]),
            input={"passage": row["passage"]},
            expected={"figures": row["expected"]},
        )
        score = figures_match(case, Outcome(output=output))
        passed += score.passed
        recall += score.value or 0.0
        all_found += score.parts.get("all_found", False)
        no_extra += score.parts.get("no_extra", False)
        described += score.parts.get("described", False)
        called = bool(prediction.get("calls"))
        if needs_call(row):
            needed += 1
            hit += called
        else:
            not_needed += 1
            false_call += called
    return Graded(
        len(rows),
        passed,
        parsed,
        recall / max(len(rows), 1),
        all_found,
        no_extra,
        described,
        hit,
        needed,
        false_call,
        not_needed,
    )


def _jsonl(path: Path) -> list[dict[str, JsonValue]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aigent.extraction.figure_eval")
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export", help="write the test set as prompts")
    exporting.add_argument("--out", type=Path, required=True)
    grading = commands.add_parser("grade", help="score a model's predictions")
    grading.add_argument("predictions", type=Path, nargs="+")
    grading.add_argument("--test", type=Path, default=None, help="the exported test set")
    rowing = commands.add_parser("rows", help="write training-format conversations as eval rows")
    rowing.add_argument("--conversations", type=Path, required=True)
    rowing.add_argument("--out", type=Path, required=True)
    rowing.add_argument("--prefix", default="va", help="the ids' prefix: va, tr")
    rowing.add_argument("--sample", type=int, default=None, help="this many rows, seeded")
    rowing.add_argument("--answers", type=Path, default=None, help="mark augmented copies")
    counting = commands.add_parser("census", help="count each kind of fault in predictions")
    counting.add_argument("predictions", type=Path)
    counting.add_argument("--test", type=Path, required=True, help="the rows they answer")
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.command == "export":
        rows = export(load_jsonl(DATASET), passages())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} cases -> {args.out}")
        return
    if args.command == "rows":
        originals = None
        if args.answers is not None:
            originals = {(str(a["passage"]), str(a["question"])) for a in _jsonl(args.answers)}
        rows = from_conversations(_jsonl(args.conversations), args.prefix, originals)
        if args.sample is not None and args.sample < len(rows):
            rows = random.Random(FIGURE_LABEL_SEED).sample(rows, args.sample)
            rows.sort(key=lambda r: str(r["id"]))
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} cases -> {args.out}")
        return
    if args.command == "census":
        rows = _jsonl(args.test)
        counts, each = census(rows, {str(p["id"]): p for p in _jsonl(args.predictions)})
        clean = sum(1 for found in each.values() if not found)
        print(f"{clean}/{len(rows)} without a fault")
        for kind, n in counts.most_common():
            print(f"  {n:4d}  {kind}")
        out = args.predictions.with_suffix(".faults.jsonl")
        out.write_text("".join(json.dumps({"id": i, "faults": f}) + "\n" for i, f in each.items()))
        print(f"faults per case -> {out}")
        return
    test = args.test or args.predictions[0].with_name("test.jsonl")
    rows = _jsonl(test)
    for path in args.predictions:
        predictions = {str(p["id"]): p for p in _jsonl(path)}
        print(f"\n{path.name}")
        for line in grade(rows, predictions).lines():
            print(f"  {line}")


if __name__ == "__main__":
    main()
