"""Template-built headlines for underhood's toy fine-tune, each label exact by construction.

uv run python -m aigent.extraction.synthetic --out ../underhood/data/toy-headlines

Each template applies one rule from `Extraction`'s field descriptions, and none copies a real
headline: the 50 labelled ones stay the test set, and a candidate sharing a four-word run with any
of them is dropped. Rows are rendered through the local wire exactly as `extraction_task` sends
them, so the model is tuned on the prompt it will be asked with.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from aigent.adapters import build, spec
from aigent.adapters.openai import OpenAI
from aigent.config import (
    SYNTHETIC_CLIENT,
    SYNTHETIC_HEADLINES,
    SYNTHETIC_SEED,
    SYNTHETIC_VALID_FRACTION,
    max_tokens,
)
from aigent.evals.dataset import load_jsonl
from aigent.extraction.headlines import (
    DATASET,
    METRICS,
    VARIANTS,
    Extraction,
    headline_request,
    load_directory,
    word_runs,
)


@dataclass(frozen=True)
class Synthetic:
    """One template headline, the record it should become, and the rule it exercises."""

    rule: str
    headline: str
    label: Extraction


UP = ("rise", "climb", "jump", "grow", "gain", "increase")
DOWN = ("fall", "drop", "decline", "slide", "dip", "shrink")
QUARTER_MONTHS = {"Q1": "June", "Q2": "September", "Q3": "December", "Q4": "March"}
BARE_MONTHS = ("October", "November", "January", "August")
WORDS = {
    "revenue": ("revenue", "net sales", "total income"),
    "profit": ("net profit", "profit", "consolidated profit"),
    "margin": ("EBITDA margin", "operating margin"),
    "other": ("deposits", "loan disbursements", "bookings", "footfalls"),
}
GROWTH_WORDS = {"revenue": "revenue growth", "other": "deposit growth"}


def mentions() -> list[str]:
    """How headlines name each company in the directory: its aliases, or its name without `Ltd`."""
    names: list[str] = []
    for entry in load_directory().values():
        names.extend(entry["aliases"] or [entry["name"].removesuffix(" Ltd")])
    return names


def _verb(rng: random.Random, word: str, up: bool) -> str:
    """A verb agreeing with its metric: net sales rise, net profit rises."""
    base = rng.choice(UP if up else DOWN)
    return base if word.endswith("s") else base + "s"


def _pct(rng: random.Random) -> float:
    return float(rng.randint(2, 45)) if rng.random() < 0.5 else round(rng.uniform(0.5, 40.0), 1)


def _level(rng: random.Random) -> float:
    return round(rng.uniform(5.0, 45.0), 1)


def _crore(rng: random.Random) -> str:
    """A rupee amount grouped the Indian way: 1,23,456 rather than 123,456."""
    digits = str(rng.randint(120, 99_999))
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "Rs " + ",".join([*groups, tail]) + " crore"


def _quarter(rng: random.Random) -> tuple[str, str]:
    """A quarter as a headline writes it, and as the label reads it."""
    quarter = rng.choice(tuple(QUARTER_MONTHS))
    if rng.random() < 0.6:
        return quarter, quarter
    return f"{QUARTER_MONTHS[quarter]} quarter", quarter


def _record(
    company: str, metric: str, quarter: str, direction: str, change_pct: float | None = None
) -> Extraction:
    return Extraction(
        company=company,
        metric=metric,
        quarter=quarter,
        direction=direction,
        change_pct=change_pct,
    )


def percent_move(rng: random.Random, company: str) -> Synthetic:
    metric = rng.choice(("revenue", "profit", "other"))
    up = rng.random() < 0.6
    pct, (said, quarter) = _pct(rng), _quarter(rng)
    tail = rng.choice(("", f" to {_crore(rng)}", " year on year"))
    word = rng.choice(WORDS[metric])
    headline = f"{company} {said} {word} {_verb(rng, word, up)} {pct:g}%{tail}"
    return Synthetic(
        "percent-move", headline, _record(company, metric, quarter, "up" if up else "down", pct)
    )


def percent_first(rng: random.Random, company: str) -> Synthetic:
    metric = rng.choice(("revenue", "profit"))
    up = rng.random() < 0.6
    pct, (said, quarter) = _pct(rng), _quarter(rng)
    move = "rise" if up else "fall"
    headline = f"{company} reports {pct:g}% {move} in {said} {rng.choice(WORDS[metric])}"
    return Synthetic(
        "percent-first", headline, _record(company, metric, quarter, "up" if up else "down", pct)
    )


def margin_level(rng: random.Random, company: str) -> Synthetic:
    """A margin moving from one level to another states no percentage change."""
    now, before = _level(rng), _level(rng)
    while now == before:
        now = _level(rng)
    up = now > before
    verb = rng.choice(("widens", "expands", "improves") if up else ("narrows", "contracts"))
    said, quarter = _quarter(rng)
    headline = f"{company} {said} {rng.choice(WORDS['margin'])} {verb} to {now:g}% from {before:g}%"
    return Synthetic(
        "margin-level", headline, _record(company, "margin", quarter, "up" if up else "down")
    )


def basis_points(rng: random.Random, company: str) -> Synthetic:
    up = rng.random() < 0.5
    said, quarter = _quarter(rng)
    unit = rng.choice(("basis points", "bps"))
    verb = rng.choice(("improves", "gains") if up else ("slips", "loses"))
    headline = (
        f"{company} {said} {rng.choice(WORDS['margin'])} {verb} {rng.randint(10, 250)} {unit}"
    )
    return Synthetic(
        "basis-points", headline, _record(company, "margin", quarter, "up" if up else "down")
    )


def rupee_amount(rng: random.Random, company: str) -> Synthetic:
    """A rupee figure, or a doubling, is not a stated percentage."""
    metric = rng.choice(("revenue", "profit"))
    up = rng.random() < 0.6
    said, quarter = _quarter(rng)
    word = rng.choice(WORDS[metric])
    if rng.random() < 0.3:
        change = ("double" if up else "halve") + ("" if word.endswith("s") else "s")
        headline = f"{company} {said} {word} {change}"
    else:
        headline = f"{company} {said} {word} {_verb(rng, word, up)} to {_crore(rng)}"
    return Synthetic(
        "rupee-amount", headline, _record(company, metric, quarter, "up" if up else "down")
    )


def flat(rng: random.Random, company: str) -> Synthetic:
    metric = rng.choice(("revenue", "profit"))
    said, quarter = _quarter(rng)
    still = rng.choice(("flat", "unchanged"))
    headline = f"{company} {said} {rng.choice(WORDS[metric])} {still} at {_crore(rng)}"
    return Synthetic("flat", headline, _record(company, metric, quarter, "flat"))


def guidance(rng: random.Random, company: str) -> Synthetic:
    """A forward-looking figure is guidance, whatever metric it is about."""
    up = rng.random() < 0.5
    verb = rng.choice(("raises", "lifts") if up else ("cuts", "lowers"))
    about = rng.choice(("revenue", "margin", "capex", "volume"))
    headline = f"{company} {verb} FY{rng.choice((26, 27))} {about} guidance"
    return Synthetic(
        "guidance", headline, _record(company, "guidance", "FY", "up" if up else "down")
    )


def orders(rng: random.Random, company: str) -> Synthetic:
    said, quarter = _quarter(rng)
    headline = f"{company} wins orders worth {_crore(rng)} in {said}"
    return Synthetic("orders", headline, _record(company, "orders", quarter, "unknown"))


def headcount(rng: random.Random, company: str) -> Synthetic:
    said, quarter = _quarter(rng)
    count = f"{rng.randint(300, 9_000):,}"
    if rng.random() < 0.5:
        headline = f"{company} adds {count} employees in {said}"
        return Synthetic("headcount", headline, _record(company, "headcount", quarter, "up"))
    headline = f"{company} {said} headcount falls by {count}"
    return Synthetic("headcount", headline, _record(company, "headcount", quarter, "down"))


def growth_rate(rng: random.Random, company: str) -> Synthetic:
    """A growth rate is a change, not a level: slower growth is still up, and its rate is the %."""
    metric = rng.choice(tuple(GROWTH_WORDS))
    now, before = sorted(rng.sample(range(3, 40), 2), reverse=rng.random() < 0.5)
    verb = rng.choice(("slows", "eases", "moderates")) if now < before else "accelerates"
    said, quarter = _quarter(rng)
    headline = f"{company} {said} {GROWTH_WORDS[metric]} {verb} to {now}% from {before}%"
    return Synthetic("growth-rate", headline, _record(company, metric, quarter, "up", float(now)))


def preferred(stated: Sequence[str], mentioned: Sequence[str]) -> str:
    """The metric a headline naming several is labelled with: the best-ranked with a stated
    percentage, else the best-ranked mentioned."""
    pool = stated or mentioned
    return min(pool, key=METRICS.index)


def two_metrics(rng: random.Random, company: str) -> Synthetic:
    said, quarter = _quarter(rng)
    first, second = rng.sample(("revenue", "profit", "other"), 2)
    moves = {metric: (rng.random() < 0.6, _pct(rng)) for metric in (first, second)}
    if rng.random() < 0.5:
        parts = [
            f"{rng.choice(WORDS[m])} {'up' if moves[m][0] else 'down'} {moves[m][1]:g}%"
            for m in (first, second)
        ]
        winner = preferred([first, second], [first, second])
        headline = f"{company} {said} {parts[0]}, {parts[1]}"
    else:
        level = f"{rng.choice(WORDS['margin'])} at {_level(rng):g}%"
        up, pct = moves[second]
        change = f"{rng.choice(WORDS[second])} {'up' if up else 'down'} {pct:g}%"
        winner = preferred([second], ["margin", second])
        headline = f"{company} {said} {level}; {change}"
    up, pct = moves[winner]
    return Synthetic(
        "two-metrics", headline, _record(company, winner, quarter, "up" if up else "down", pct)
    )


def share_price(rng: random.Random, company: str) -> Synthetic:
    """A share price move is the market's number: no metric, and its % is no change_pct."""
    move = rng.choice(("shares jump", "shares slump", "stock rises", "stock falls"))
    tail = rng.choice(("in early trade", "after a block deal", "on brokerage upgrade"))
    headline = f"{company} {move} {_pct(rng):g}% {tail}"
    return Synthetic("share-price", headline, _record(company, "none", "unknown", "unknown"))


def no_metric(rng: random.Random, company: str) -> Synthetic:
    event = rng.choice(
        (
            "appoints new chief executive",
            "names new finance chief",
            f"board to meet on {rng.choice(BARE_MONTHS)} {rng.randint(2, 28)}",
        )
    )
    return Synthetic(
        "no-metric", f"{company} {event}", _record(company, "none", "unknown", "unknown")
    )


def no_period(rng: random.Random, company: str) -> Synthetic:
    """No period named, or only a bare month, which is not a quarter."""
    metric = rng.choice(("revenue", "profit", "other"))
    up = rng.random() < 0.6
    pct = _pct(rng)
    month = f" in {rng.choice(BARE_MONTHS)}" if rng.random() < 0.5 else ""
    word = rng.choice(WORDS[metric])
    headline = f"{company} {word} {_verb(rng, word, up)} {pct:g}%{month}"
    return Synthetic(
        "no-period", headline, _record(company, metric, "unknown", "up" if up else "down", pct)
    )


def half_or_full_year(rng: random.Random, company: str) -> Synthetic:
    metric = rng.choice(("revenue", "profit"))
    up = rng.random() < 0.6
    pct = _pct(rng)
    said, period = rng.choice(
        (("H1", "H1"), ("first-half", "H1"), ("H2", "H2"), ("second-half", "H2"), ("FY25", "FY"))
    )
    word = rng.choice(WORDS[metric])
    headline = f"{company} {said} {word} {_verb(rng, word, up)} {pct:g}%"
    return Synthetic(
        "half-or-full-year", headline, _record(company, metric, period, "up" if up else "down", pct)
    )


def loss(rng: random.Random, company: str) -> Synthetic:
    """A narrowing loss is profit going up."""
    narrows = rng.random() < 0.5
    said, quarter = _quarter(rng)
    headline = f"{company} {said} net loss {'narrows' if narrows else 'widens'} to {_crore(rng)}"
    return Synthetic(
        "loss", headline, _record(company, "profit", quarter, "up" if narrows else "down")
    )


Template = Callable[[random.Random, str], Synthetic]

TEMPLATES: tuple[Template, ...] = (
    percent_move,
    percent_first,
    margin_level,
    basis_points,
    rupee_amount,
    flat,
    guidance,
    orders,
    headcount,
    growth_rate,
    two_metrics,
    share_price,
    no_metric,
    no_period,
    half_or_full_year,
    loss,
)


def real_headlines() -> list[str]:
    return [str(case.input["headline"]) for case in load_jsonl(DATASET)]


def generate(
    n: int = SYNTHETIC_HEADLINES, seed: int = SYNTHETIC_SEED, real: Sequence[str] | None = None
) -> list[Synthetic]:
    """`n` distinct headlines, the templates taken in turn, none quoting a real headline."""
    rng = random.Random(seed)
    companies = mentions()
    banned = set[tuple[str, ...]]().union(*(word_runs(h) for h in (real or real_headlines())))
    seen: set[str] = set()
    made: list[Synthetic] = []
    for attempt in range(50 * n):
        if len(made) == n:
            break
        example = TEMPLATES[attempt % len(TEMPLATES)](rng, rng.choice(companies))
        key = example.headline.casefold()
        if key in seen or word_runs(example.headline) & banned:
            continue
        seen.add(key)
        made.append(example)
    if len(made) < n:
        raise RuntimeError(f"made only {len(made)} of {n} headlines; the templates are too narrow")
    return made


def split(
    examples: Sequence[Synthetic], valid_fraction: float, seed: int
) -> tuple[list[Synthetic], list[Synthetic]]:
    """Shuffled once by `seed`, the last share kept back for validation."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    cut = len(shuffled) - max(1, round(len(shuffled) * valid_fraction))
    return shuffled[:cut], shuffled[cut:]


def wire_for(client: str) -> OpenAI:
    """The client's wire, which must be the OpenAI chat format mlx_lm trains on."""
    adapter = build(client)
    if not isinstance(adapter, OpenAI):
        raise ValueError(
            f"{client} speaks {spec(client).wire}; mlx_lm trains on the openai chat format"
        )
    return adapter


def render(example: Synthetic, wire: OpenAI, cap: int) -> dict[str, list[dict[str, object]]]:
    """One training row: the request `extraction_task` would send, then the label as the reply."""
    request = headline_request("synthetic", example.headline, VARIANTS["zero_shot"], cap)
    messages: list[dict[str, object]] = [dict(m) for m in wire.messages(request, Extraction)]
    messages.append({"role": "assistant", "content": json.dumps(example.label.model_dump())})
    return {"messages": messages}


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aigent.extraction.synthetic",
        description="Write template headlines as mlx_lm training rows. Free: no calls.",
    )
    parser.add_argument("--out", type=Path, required=True, help="directory for train/valid.jsonl")
    parser.add_argument("--n", type=int, default=SYNTHETIC_HEADLINES, help="how many headlines")
    parser.add_argument(
        "--seed", type=int, default=SYNTHETIC_SEED, help="what makes a set repeatable"
    )
    parser.add_argument("--client", default=SYNTHETIC_CLIENT, help="whose wire to render for")
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    examples = generate(args.n, args.seed)
    wire, cap = wire_for(args.client), max_tokens("HEADLINE", spec(args.client))
    train, valid = split(examples, SYNTHETIC_VALID_FRACTION, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        lines = (json.dumps(render(row, wire, cap), ensure_ascii=False) for row in rows)
        (args.out / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: {len(rows)} rows -> {args.out / f'{name}.jsonl'}")
    for rule, count in sorted(Counter(example.rule for example in examples).items()):
        print(f"  {rule}: {count}")


if __name__ == "__main__":
    main()
