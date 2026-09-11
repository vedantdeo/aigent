"""Week 1, Project 1a: the first eval with real money behind it.

    uv run python -m entropic.week01.extraction --yes

Everything up to here was a primitive. This is the first thing shaped like a product: a schema, a
prompt, a hand-labelled dataset, and a number that says whether a prompt change helped.

What to notice:
  - **The model names the company; code resolves the ticker.** The model returns whatever the
    headline says — "Infy", "HUL", "L&T" — and `Resolver` maps that to an NSE symbol through the
    directory. A dict lookup is exact, free, and cannot hallucinate `HEROMOTOCORP` for
    `HEROMOTOCO`. Asking a frontier model to do a join is paying frontier rates for `dict.get`.
    Eleven of the twenty-nine tickers here are not derivable from any name the headline uses, so
    this is most of the field's difficulty removed by construction rather than by prompting.
  - **A mention the directory does not know is a gap in the directory, not a wrong answer.** It
    resolves to `None`, the row fails honestly, and `main` prints the mentions to add. The failure
    mode of a lookup is a to-do list; the failure mode of a guess is a plausible wrong symbol.
  - **The task owns its call.** `extraction_task` returns a `Task`, which is `Case -> Outcome`. The
    runner never learns that an API is involved, which is why the same runner grades Week 2's
    retrieval function for free. Resolution happens inside the task, so the harness needs no
    concept of post-processing either.
  - **A refusal to parse is an `Outcome` with an error, not an exception.** Same rule as tool
    errors: one bad row is data about the system, not the end of the run.
  - **Two variants, one dataset.** A single score is unreadable — 71% of what? Two prompts over the
    same 30 rows is a comparison, and the per-field table says which field the difference is in.
  - **The output cap is sized for the answer.** `MAX_TOKENS_HEADLINE` is 384, not the 2048 the
    paper-summary demo uses; the pre-flight guard prices the full cap, so an oversized one makes a
    cheap eval look unaffordable — at 2048 this run's worst case is $3.19 against a $2.00 ceiling.
    It started at 256 and that clipped one row in 60 on the first live run, which is the other edge
    of the same knife: too tight loses rows, too loose refuses the run.
  - **A tie-break the model can apply beats a judgment call it has to guess.** `METRICS` is a total
    order, and the `metric` description says exactly how to use it. A headline naming two metrics
    with no stated rule produces a label that is a coin flip, and the miss reads as a model failure
    when it is a spec failure.
  - **The conventions live in the field descriptions**, which the model sees. Few-shot then shows
    them being applied. If a rule only ever appears in the examples, the zero-shot arm is being
    punished for a rule nobody told it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

import anthropic
from anthropic.types import MessageParam
from pydantic import BaseModel, Field

from entropic.config import MAX_TOKENS_HEADLINE as MAX_TOKENS
from entropic.config import MAX_USD_PER_EVAL, MODEL, get_client
from entropic.evals.dataset import Case, digest, load_jsonl
from entropic.evals.grade import Outcome, field_match, pydantic_valid
from entropic.evals.report import write_report
from entropic.evals.runner import Task, run_eval
from entropic.pricing import check_request, estimate_eval_usd

REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "evals" / "datasets" / "headlines.jsonl"
DIRECTORY = REPO / "evals" / "reference" / "nse-tickers.json"

# What gets graded. `ticker` is resolved in code; the other four are the model's judgment.
FIELDS = ("ticker", "metric", "quarter", "direction", "change_pct")

# A total order over the metric vocabulary, most preferred first. A headline routinely mentions two
# ("narrows Q1 loss; revenue up 63%") and without a tie-break the label is a coin flip that the
# model has no way to call — which shows up as a metric failure that is really a spec failure.
#
# PLACEHOLDER. The order is asserted, not researched: top line, then bottom line, then the ratio
# between them, then the forward-looking and operational signals. It removes the ambiguity, which is
# its whole job right now; replace it with something grounded in how these headlines are actually
# read before drawing conclusions about which metric a model is worst at.
METRICS = ("revenue", "profit", "margin", "orders", "headcount", "guidance")


class Extraction(BaseModel):
    """One results headline as a record. Descriptions are visible to the model — write them as
    prompts, because that is what they are."""

    company: str = Field(
        description=(
            "The company the numbers belong to, copied verbatim from the headline. Do not expand "
            "it, canonicalise it, or convert it to a ticker: if the headline says Infy, write "
            "Infy; if it says HUL, write HUL. If two companies appear, pick the one the numbers "
            "describe."
        )
    )
    metric: str = Field(
        description=(
            "Exactly one of, in this order of preference: " + ", ".join(METRICS) + ". A headline "
            "often touches more than one. Choose the metric earliest in that list among the ones "
            "whose percentage change the headline actually states; if it states no percentage "
            "change for any of them, choose the one earliest in the list overall. A "
            "forward-looking statement about a metric is guidance, not that metric: 'cuts revenue "
            "guidance' and 'guides for revenue growth' are both guidance, and name only one "
            "metric."
        )
    )
    quarter: str = Field(
        description=(
            "Exactly one of: Q1, Q2, Q3, Q4, FY, unknown. Indian fiscal year, so the June quarter "
            "is Q1, September is Q2, December is Q3 and March is Q4. Use unknown when the "
            "headline names no period."
        )
    )
    direction: str = Field(
        description="Exactly one of: up, down, flat, unknown. Which way the metric moved."
    )
    change_pct: float | None = Field(
        default=None,
        description=(
            "The percentage change, as a positive number, only when the headline states one "
            "explicitly. Null otherwise — a level ('margin to 3.6% from 3.4%'), a move in basis "
            "points, a rupee figure, or a word like 'doubles' are all null."
        ),
    )


class Company(TypedDict):
    """One row of the ticker directory."""

    name: str
    aliases: list[str]


def load_directory(path: Path = DIRECTORY) -> dict[str, Company]:
    """Ticker -> registered name and the forms a headline actually uses.

    Reference data, not settings. It grounds the dataset's `ticker` labels — a label outside it is
    unfalsifiable — and it is what `Resolver` looks up.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Company], raw["companies"])


def _key(mention: str) -> str:
    """Normalise a company mention for lookup: case, punctuation, spacing and the Ltd suffix.

    Deliberately not fuzzy. A lookup that matches approximately can be wrong in the same quiet way
    a model can, and then neither half of the system is trustworthy.
    """
    text = re.sub(r"\b(ltd|limited|inc|plc)\b\.?", "", mention.casefold())
    return re.sub(r"[^a-z0-9&]", "", text)


@dataclass
class Resolver:
    """Company mention -> NSE ticker, through the directory. Records what it could not resolve.

    `unresolved` is the deliverable when this fails: a counted list of mentions to add to
    `evals/reference/nse-tickers.json`, rather than a set of rows that are wrong for no stated
    reason.
    """

    directory: dict[str, Company] = field(default_factory=load_directory)
    unresolved: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self._index: dict[str, str] = {}
        for ticker, entry in self.directory.items():
            for name in (ticker, entry["name"], *entry["aliases"]):
                self._index.setdefault(_key(name), ticker)

    def __call__(self, mention: str) -> str | None:
        ticker = self._index.get(_key(mention))
        if ticker is None:
            self.unresolved[mention] += 1
        return ticker


ZERO_SHOT = (
    "Extract one structured record from an Indian market results headline. Follow each field's "
    "description exactly. When the headline does not state something, say so with unknown or null "
    "rather than inferring it."
)

FEW_SHOT = ZERO_SHOT + (
    "\n\nWorked examples:\n\n"
    "headline: Infy Q1 revenue up 4.2%\n"
    '{"company": "Infy", "metric": "revenue", "quarter": "Q1", "direction": "up", '
    '"change_pct": 4.2}\n\n'
    "headline: Nestle India March quarter margin widens to 22.1% from 21.4%\n"
    '{"company": "Nestle India", "metric": "margin", "quarter": "Q4", "direction": "up", '
    '"change_pct": null}\n\n'
    "headline: Tata Steel cuts capex plans for the year\n"
    '{"company": "Tata Steel", "metric": "guidance", "quarter": "FY", "direction": "down", '
    '"change_pct": null}\n\n'
    "headline: Britannia Q2 margin widens to 17.2%; revenue up 4%\n"
    '{"company": "Britannia", "metric": "revenue", "quarter": "Q2", "direction": "up", '
    '"change_pct": 4.0}\n\n'
    "The first answer is Infy, not Infosys: copy what the headline wrote. The second states two "
    "percentages and change_pct is still null — they are levels, not a change. The third names no "
    "quarter but does name the year, so it is FY. The fourth mentions two metrics and only revenue "
    "has a stated percentage change, so revenue wins even though margin leads the headline."
)

# Each arm differs from the one before it by exactly one thing, which is what makes the per-field
# table read as an ablation rather than a scatter.
VARIANTS = {"zero_shot": ZERO_SHOT, "few_shot": FEW_SHOT}


def extraction_task(
    system: str,
    resolve: Resolver | None = None,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> Task:
    """Build the `Task` the runner calls once per case. `client` is injectable so tests are free."""
    built = client
    resolver = resolve if resolve is not None else Resolver()

    def task(case: Case) -> Outcome:
        nonlocal built
        if built is None:
            built = get_client()
        headline = case.input.get("headline")
        if not isinstance(headline, str):
            return Outcome(error=f"case {case.id} has no headline")

        messages: list[MessageParam] = [{"role": "user", "content": f"headline: {headline}"}]
        check_request(built, model=model, max_tokens=MAX_TOKENS, messages=messages, system=system)
        response = built.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            output_format=Extraction,
        )

        record = response.parsed_output
        if record is None:
            return Outcome(
                usage=response.usage,
                model=model,
                error=f"no parsed output; stop_reason={response.stop_reason}",
            )
        # The model's record travels too, under its own key: `pydantic_valid` still sees the shape
        # the model was asked for, and a failed row shows the mention that did not resolve.
        output = record.model_dump()
        output["ticker"] = resolver(record.company)
        return Outcome(output=output, usage=response.usage, model=model)

    return task


def _rough_input_tokens(system: str) -> int:
    """Four characters to the token, plus room for the headline. Deliberately crude: this is a
    pre-flight estimate, and `check_request` counts the real thing before each call anyway."""
    return len(system) // 4 + 40


def spread(cases: Sequence[Case], n: int) -> list[Case]:
    """`n` cases taken evenly across the dataset, not the first `n`.

    A smoke run is only worth its money if it can fail. The dataset is ordered roughly by how it was
    written, so the first four rows are four easy ones; every k-th row spans the traps.
    """
    if n >= len(cases):
        return list(cases)
    return list(cases[:: max(1, len(cases) // n)])[:n]


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="entropic.week01.extraction",
        description="Project 1a: structured extraction over hand-labelled headlines.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="send the calls; without it, estimate only"
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="run N cases spread across the dataset, for a cheap smoke run before the whole thing",
    )
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    cases = load_jsonl(DATASET)
    labelled = [case for case in cases if case.expected]
    skipped = len(cases) - len(labelled)
    if args.sample is not None:
        labelled = spread(labelled, args.sample)

    # Price each arm by its own prompt rather than averaging the two.
    worst = sum(
        estimate_eval_usd(MODEL, len(labelled), _rough_input_tokens(system), MAX_TOKENS)
        for system in VARIANTS.values()
    )
    sampled = f", sampled to {len(labelled)}" if args.sample is not None else ""
    print(
        f"{DATASET.name}: {len(labelled)} labelled cases{sampled}, {skipped} unlabelled and skipped"
    )
    print(f"variants: {', '.join(VARIANTS)}  on {MODEL}")
    print(f"directory: {len(load_directory())} companies")
    print(f"worst case ${worst:.2f} against a ${MAX_USD_PER_EVAL:.2f} ceiling")

    if not args.yes:
        print("\nnothing spent. re-run with --yes to send these calls.")
        return

    resolver = Resolver()
    run = run_eval(
        labelled,
        {name: extraction_task(system, resolver) for name, system in VARIANTS.items()},
        {"valid": pydantic_valid(Extraction), "fields": field_match(FIELDS)},
        dataset=DATASET.name,
        digest=digest(DATASET),
        model=MODEL,
    )
    print(f"\nreport: {write_report(run)}")
    report_unresolved(resolver)


def report_unresolved(resolver: Resolver) -> None:
    """The actionable half of a failed lookup: what to add to the directory, and how often it came
    up. Silence here means every mention the model produced was one the directory knows."""
    if not resolver.unresolved:
        print("directory: every company mention resolved")
        return
    print(f"\ndirectory: {len(resolver.unresolved)} mentions did not resolve — add them to")
    print(f"  {DIRECTORY.relative_to(REPO)}")
    for mention, seen in resolver.unresolved.most_common():
        print(f'    "{mention}"  ({seen}x)')


if __name__ == "__main__":
    main()
