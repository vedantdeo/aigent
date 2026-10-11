"""The figure fine-tune's eval: the hand-labelled test set as prompts a model answers elsewhere, and
the grade of what it answered, figures and tool use alike.

uv run python -m aigent.extraction.figure_eval export --out ../underhood/data/figures/test.jsonl
uv run python -m aigent.extraction.figure_eval grade ../underhood/data/figures/eval-tuned.jsonl

A prediction is one JSON line: {"id", "final" (the last reply), "calls" (each calculate expression
the model made), "error" (or null)}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from aigent.evals.dataset import Case, load_jsonl
from aigent.evals.grade import Outcome
from aigent.extraction.figure_training import source_of
from aigent.extraction.figures import (
    DATASET,
    TOOL,
    Figures,
    _printed,
    figure_messages,
    figures_match,
    parse_figures,
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


def needs_call(case_row: dict[str, JsonValue]) -> bool:
    """Whether answering a case takes a calculate call: some expected value is not printed as is."""
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
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.command == "export":
        rows = export(load_jsonl(DATASET), passages())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} cases -> {args.out}")
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
