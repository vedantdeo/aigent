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
    order and the `metric` description says how to use it. A headline naming two metrics with no
    stated rule produces a label that is a coin flip, and the miss reads as a model failure when it
    is a spec failure.
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
from anthropic.types import MessageParam, TextBlockParam, ThinkingConfigParam
from pydantic import BaseModel, Field

from entropic.config import MAX_TOKENS_HEADLINE as MAX_TOKENS
from entropic.config import MAX_USD_PER_EVAL, MODEL, THINKING_EVAL, get_client
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

# Tie-break order for `metric`, most preferred first. Placeholder: asserted, not researched.
METRICS = ("revenue", "profit", "margin", "orders", "headcount", "guidance", "other", "none")


class Extraction(BaseModel):
    """One results headline as a record. Descriptions are visible to the model — write them as
    prompts, because that is what they are, and keep them short for the same reason."""

    company: str | None = Field(
        description=(
            "Copied verbatim from the headline: Infy stays Infy, HUL stays HUL, never a ticker. "
            "An unlisted subsidiary is the one exception — take its listed parent (Jaguar Land "
            "Rover is Tata Motors). Two companies: the one the numbers describe, else the one "
            "named first. Null when no single company is named, as for a sector or an index."
        )
    )
    metric: str = Field(
        description=(
            "Exactly one of, in this order of preference: " + ", ".join(METRICS) + ". Take the "
            "earliest ranked among the metrics whose percentage change the headline states; "
            "failing that, the earliest among those it mentions at all; failing that, the one "
            "mentioned first. other is a company metric the six do not name — footfalls, "
            "bookings, deposits, loan disbursements — not the nearest of the six. none is a "
            "headline reporting no metric. A forward-looking statement is guidance, not that "
            "metric: 'cuts revenue guidance' is guidance. A share price move is the market's "
            "number, never the metric, and its percentage is never change_pct."
        )
    )
    quarter: str = Field(
        description=(
            "Exactly one of: Q1, Q2, Q3, Q4, H1, H2, FY, unknown. Indian fiscal year, so the "
            "quarters ending June, September, December and March are Q1 to Q4. A bare month is "
            "not a quarter: 'in October' is unknown, as is a headline naming no period."
        )
    )
    direction: str = Field(
        description=(
            "Exactly one of: up, down, flat, unknown. Which way the metric itself moved. For a "
            "rate of growth that is the sign of the growth, not whether the rate rose or fell: "
            "'deposit growth eases to 4% from 7%' is up. unknown when no direction is stated, or "
            "there is no metric to have one."
        )
    )
    change_pct: float | None = Field(
        default=None,
        description=(
            "The percentage change, positive, only when the headline states one explicitly. Null "
            "otherwise: a level ('margin to 19.4% from 18.1%'), basis points, a rupee figure, "
            "'doubles'. A growth rate is already a change and not a level, so 'deposit growth "
            "eases to 4% from 7%' is 4."
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
    "headline: Tata Motors' Jaguar Land Rover Q3 revenue up 11%\n"
    '{"company": "Tata Motors", "metric": "revenue", "quarter": "Q3", "direction": "up", '
    '"change_pct": 11.0}\n\n'
    "headline: Titan Q2 footfall growth eases to 5% from 9%\n"
    '{"company": "Titan", "metric": "other", "quarter": "Q2", "direction": "up", '
    '"change_pct": 5.0}\n\n'
    "Two of these are easy to read backwards. In the fourth, only revenue has a stated percentage "
    "change, so it wins even though margin leads. In the sixth, growth that slowed is still "
    "growth, so the direction is up."
)


# Turning THINKING_EVAL on means raising MAX_TOKENS_HEADLINE too: the budget below has to fit
# under it with room left for the answer, and 128 is sized for the record alone.
THINKING: ThinkingConfigParam = (
    {"type": "enabled", "budget_tokens": 1024} if THINKING_EVAL else {"type": "disabled"}
)


@dataclass(frozen=True)
class Variant:
    """One arm of an eval: the system prompt, and whether it travels behind a cache breakpoint.

    The two axes are independent on purpose. Prompt *content* changes what the model answers;
    a breakpoint changes only what the answer costs, so holding one fixed while moving the other
    is what makes either table readable.
    """

    system: str
    cache: bool = False

    def as_sent(self) -> str | list[TextBlockParam]:
        """What goes on the wire. A breakpoint needs the block form; a plain prompt does not."""
        if not self.cache:
            return self.system
        return [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]


# Each arm adds exactly one thing to the last, so the per-field table reads as an ablation.
VARIANTS = {"zero_shot": Variant(ZERO_SHOT), "few_shot": Variant(FEW_SHOT)}

# The caching arms hold the prompt fixed and move only the breakpoint. Scores must come out
# identical — the cache is a serving detail, not a different request — so a gap here is a bug.
CACHE_VARIANTS = {"few_shot": Variant(FEW_SHOT), "few_shot_cached": Variant(FEW_SHOT, cache=True)}


def extraction_task(
    variant: Variant,
    resolve: Resolver | None = None,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> Task:
    """Build the `Task` the runner calls once per case. `client` is injectable so tests are free."""
    built = client
    resolver = resolve if resolve is not None else Resolver()
    system = variant.as_sent()

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
            thinking=THINKING,
            output_format=Extraction,
        )

        record = response.parsed_output
        if record is None:
            return Outcome(
                usage=response.usage,
                model=model,
                error=f"no parsed output; stop_reason={response.stop_reason}",
            )
        # The mention travels too, so `pydantic_valid` sees the model's own shape.
        output = record.model_dump()
        output["ticker"] = resolver(record.company) if record.company else None
        return Outcome(output=output, usage=response.usage, model=model)

    return task


@dataclass(frozen=True)
class Shape:
    """How big one row's request is, and how much of a breakpoint would cover."""

    total: int
    cached: int


def measure(
    client: anthropic.Anthropic, variant: Variant, headline: str, model: str = MODEL
) -> Shape:
    """Count one row's request, for free, before deciding to pay for fifty of them.

    This used to divide the system prompt by four, which never saw `output_format` — and the
    schema is the largest fixed part of every request here, four times `ZERO_SHOT`. The estimate
    ran 37% low for the whole of Project 1a. Counting is not billed, so there was never a reason
    to guess.
    """
    probe: list[MessageParam] = [{"role": "user", "content": f"headline: {headline}"}]
    total = client.messages.count_tokens(
        model=model, system=variant.as_sent(), messages=probe, output_format=Extraction
    ).input_tokens
    if not variant.cache:
        return Shape(total, cached=0)
    # Everything ahead of the breakpoint is cached; only the headline is resent.
    tail = client.messages.count_tokens(model=model, messages=probe).input_tokens
    return Shape(total, cached=total - tail)


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
    parser.add_argument(
        "--cache",
        action="store_true",
        help="swap the prompt ablation for the caching one: one prompt, breakpoint off then on",
    )
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    cases = load_jsonl(DATASET)
    labelled = [case for case in cases if case.expected]
    skipped = len(cases) - len(labelled)
    if args.sample is not None:
        labelled = spread(labelled, args.sample)

    variants = CACHE_VARIANTS if args.cache else VARIANTS
    sampled = f", sampled to {len(labelled)}" if args.sample is not None else ""
    print(
        f"{DATASET.name}: {len(labelled)} labelled cases{sampled}, {skipped} unlabelled and skipped"
    )
    print(f"variants: {', '.join(variants)}  on {MODEL}")
    print(f"directory: {len(load_directory())} companies")

    # Counting is free, so price each arm off the real request rather than a guess at it.
    client = get_client()
    longest = max((str(case.input["headline"]) for case in labelled), key=len, default="")
    worst = 0.0
    for name, variant in variants.items():
        shape = measure(client, variant, longest)
        worst += estimate_eval_usd(
            MODEL, len(labelled), shape.total, MAX_TOKENS, cached_tokens=shape.cached
        )
        cached = f", {shape.cached} of them cached" if shape.cached else ""
        print(f"  {name}: {shape.total} input tokens per row{cached}")
    print(f"worst case ${worst:.2f} against a ${MAX_USD_PER_EVAL:.2f} ceiling")

    if not args.yes:
        print("\nnothing spent. re-run with --yes to send these calls.")
        return

    resolver = Resolver()
    run = run_eval(
        labelled,
        {n: extraction_task(v, resolver, client=client) for n, v in variants.items()},
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
