"""The headline reader's eval for a model answered elsewhere: the labelled headlines as the prompts
the training rows were rendered from, and a graded report of the replies, by Project 1a's graders.

uv run python -m aigent.extraction.headline_eval export --out test.jsonl
uv run python -m aigent.extraction.headline_eval grade predictions.jsonl --name qwen3-4b-qlora

A prediction is one JSON line, as underhood's tool_eval writes it: {"id", "final", "error"}. Free.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from aigent.config import SYNTHETIC_CLIENT, max_tokens
from aigent.evals.dataset import Case, digest, load_jsonl
from aigent.evals.grade import Outcome, field_match, pydantic_valid, unparsed
from aigent.evals.report import write_report
from aigent.evals.runner import Task, run_eval
from aigent.extraction.headlines import (
    DATASET,
    FIELDS,
    VARIANTS,
    Extraction,
    Resolver,
    headline_request,
)
from aigent.extraction.synthetic import wire_for
from aigent.llm.adapters import spec
from aigent.llm.adapters.openai import record_in


def export(cases: Sequence[Case], client: str = SYNTHETIC_CLIENT) -> list[dict[str, JsonValue]]:
    """Each labelled case as {"id", "messages"}: the zero-shot request on the client's wire, exactly
    as `synthetic.render` writes a training row, less the reply."""
    wire, cap = wire_for(client), max_tokens("HEADLINE", spec(client))
    rows: list[dict[str, JsonValue]] = []
    for case in cases:
        if not case.expected:
            continue
        request = headline_request(case.id, str(case.input["headline"]), VARIANTS["zero_shot"], cap)
        messages = cast(list[JsonValue], [dict(m) for m in wire.messages(request, Extraction)])
        rows.append({"id": case.id, "messages": messages})
    return rows


def replay_task(
    predictions: Mapping[str, Mapping[str, JsonValue]], name: str, resolve: Resolver
) -> Task:
    """A `Task` answering each case from its saved reply, read and resolved as `extraction_task`
    reads a live one; a case with no reply, or one that failed, is an error."""

    def task(case: Case) -> Outcome:
        prediction = predictions.get(case.id)
        if prediction is None:
            return Outcome(model=name, error=f"no prediction for {case.id}")
        final = prediction.get("final")
        if not isinstance(final, str):
            return Outcome(model=name, error=str(prediction.get("error") or "no final reply"))
        record = record_in(final, Extraction)
        if record is None:
            return Outcome(model=name, raw=final, error=unparsed(None, final))
        output: dict[str, JsonValue] = record.model_dump()
        output["ticker"] = resolve(record.company) if record.company else None
        return Outcome(output=output, model=name, raw=final)

    return task


def _jsonl(path: Path) -> list[dict[str, JsonValue]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aigent.extraction.headline_eval")
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export", help="write the labelled headlines as prompts")
    exporting.add_argument("--out", type=Path, required=True)
    grading = commands.add_parser("grade", help="grade a model's saved replies")
    grading.add_argument("predictions", type=Path)
    grading.add_argument("--name", required=True, help="the model, as the report names it")
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    cases = load_jsonl(DATASET)
    if args.command == "export":
        rows = export(cases)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lines = (json.dumps(row, ensure_ascii=False) for row in rows)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{len(rows)} prompts -> {args.out}")
        return
    predictions = {str(p["id"]): p for p in _jsonl(args.predictions)}
    run = run_eval(
        [case for case in cases if case.expected],
        {"zero_shot": replay_task(predictions, args.name, Resolver())},
        {"valid": pydantic_valid(Extraction), "fields": field_match(FIELDS)},
        dataset=DATASET.name,
        digest=digest(DATASET),
        model=args.name,
        worst_usd=0.0,
    )
    print(f"report: {write_report(run)}")


if __name__ == "__main__":
    main()
