"""Week 1, Project 1a: structured extraction over labelled headlines, with an eval behind it.

uv run python -m entropic.extraction.headlines --yes

The model names the company as the headline writes it and `Resolver` maps that to an NSE symbol; a
mention the directory does not know resolves to None and is reported as a gap in the directory. Two
prompt variants over one dataset, so the score is a comparison rather than a bare number.
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
    """Read the ticker directory: symbol -> registered name plus the aliases headlines use."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Company], raw["companies"])


def _key(mention: str) -> str:
    """Normalised form a company mention is indexed and looked up by."""
    text = re.sub(r"\b(ltd|limited|inc|plc)\b\.?", "", mention.casefold())
    return re.sub(r"[^a-z0-9&]", "", text)


@dataclass
class Resolver:
    """Company mention -> NSE symbol, by dictionary lookup.

    Indexes with `setdefault`, so two companies claiming one spelling is a collision worth a test.
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
    """One arm of the eval: a name, a system prompt, and whether to cache it."""

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
    """Count the tokens one variant's request really sends, through the free endpoint.

    Counting the real request rather than estimating it: the schema alone is over a thousand tokens.
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
    """Per-field pass rates for one variant, for the line `main` prints after a run."""
    if n >= len(cases):
        return list(cases)
    return list(cases[:: max(1, len(cases) // n)])[:n]


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="entropic.extraction.headlines",
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
