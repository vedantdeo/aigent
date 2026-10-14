"""The specialists' contract and each one's check: a narrow job a fine-tune answers on the prompt it
was trained on, with the generalist as the other model that can do it. `analyst.router` runs it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import JsonValue
from underhood.tooldata.labels import Label, check_call, printed

from aigent.extraction.figures import CHANGE, Figures, parse_figures
from aigent.extraction.headlines import DIRECTIONS, METRICS, QUARTERS, Extraction, Resolver
from aigent.llm.adapters.openai import record_in


class Invalid(ValueError):
    """An answer its specialist's check refused, and why."""


@dataclass(frozen=True)
class Told:
    """One model's reply to a job, what it cost, and the calculate calls it made on the way."""

    text: str
    usd: float
    input_tokens: int
    output_tokens: int
    calls: tuple[str, ...] = ()


@dataclass(frozen=True)
class Specialist[Job, Out]:
    """A narrow job two models can do: `ask` puts it to the fine-tune and raises
    `SpecialistUnreachable` if it cannot, `fallback` puts it to the generalist, and `check` turns
    either reply into the answer or raises `Invalid`."""

    name: str
    ask: Callable[[Job], Told]
    fallback: Callable[[Job], Told]
    check: Callable[[Job, Told], Out]


def check_headline(resolve: Resolver) -> Callable[[str, Told], dict[str, JsonValue]]:
    """The headline reader's check: a record in the three vocabularies, its percentage printed in
    the headline, and its company, if it names one, resolved to a ticker the record then carries."""

    def check(headline: str, told: Told) -> dict[str, JsonValue]:
        record = record_in(told.text, Extraction)
        if record is None:
            raise Invalid("the reply holds no headline record")
        named = (
            ("metric", record.metric, METRICS),
            ("quarter", record.quarter, QUARTERS),
            ("direction", record.direction, DIRECTIONS),
        )
        for name, value, vocabulary in named:
            if value not in vocabulary:
                raise Invalid(f"{name} {value!r} is not one of {', '.join(vocabulary)}")
        if record.change_pct is not None and abs(record.change_pct) not in printed(headline):
            raise Invalid(f"the headline prints no {record.change_pct}")
        ticker = resolve(record.company) if record.company else None
        if record.company and ticker is None:
            raise Invalid(f"{record.company!r} is not a company the directory lists")
        return {**record.model_dump(), "ticker": ticker}

    return check


@dataclass(frozen=True)
class FigureJob:
    """What the report analyst is asked: a passage, the report it is from, and what to extract."""

    passage: str
    source: str
    wanted: str


_SCALED = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*\*\s*(?:10\s*\*\*\s*(\d+)|1e(\d+)|1(0{3,}))\s*\)?")


def canonical(call: str) -> str:
    """A scaling call in the one form the check reads, `N * 10**k`: `N * 10000000`, `N*1e7` and
    `(N * 10**7)` all become `N * 10**7`. Any other call is returned as it came."""
    found = _SCALED.fullmatch(call.strip())
    if found is None:
        return call
    number, power, exponent, zeros = found.groups()
    return f"{number} * 10**{power or exponent or len(zeros)}"


def check_figures(job: FigureJob, told: Told) -> Figures:
    """The report analyst's check, all or nothing: the reply is `Figures`, and every value is a
    number the passage prints, as printed or through one of the reply's calculate calls."""
    figures = parse_figures(told.text)
    if figures is None:
        raise Invalid("the reply holds no figures")
    for figure in figures.figures:
        backed = any(
            check_call(Label(**figure.model_dump(), call=call), job.passage, CHANGE)
            for call in (None, *(canonical(c) for c in told.calls))
        )
        if not backed:
            raise Invalid(f"the passage does not back {figure.value} for {figure.metric}")
    return figures
