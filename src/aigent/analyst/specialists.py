"""The specialists' contract and the report analyst's check: a narrow job a fine-tune answers on
the prompt it was trained on, with the generalist as the other model that can do it, and the
passages a job read. `analyst.router` runs it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from underhood.tooldata.labels import Label, check_call

from aigent.extraction.figures import CHANGE
from aigent.extraction.research import BASES, SourcedFigure, parse_sourced
from aigent.llm.messages import Msg
from aigent.retrieval.chunk import squeeze

_PASSAGE = re.compile(r'<passage id="([^"]+)" source="([^"]*)">\n(.*?)\n</passage>', re.S)


@dataclass(frozen=True)
class Passage:
    """One passage a search returned: the id it is cited by, where it is from, and its text."""

    id: str
    source: str
    text: str

    @property
    def report(self) -> str:
        """The report it is from: ITC-FY25 for ITC-FY25#0412."""
        return self.id.split("#")[0]


def passages_in(result: str) -> list[Passage]:
    """Every passage in a search result's text, in order."""
    return [Passage(*found) for found in _PASSAGE.findall(result)]


class Invalid(ValueError):
    """An answer its specialist's check refused, and why."""


@dataclass(frozen=True)
class Told:
    """One model's reply to a job, what it cost, the calculate calls it made on the way, the
    passages its searches returned, and the whole conversation from the first user turn on."""

    text: str
    usd: float
    input_tokens: int
    output_tokens: int
    calls: tuple[str, ...] = ()
    read: tuple[Passage, ...] = ()
    conversation: tuple[Msg, ...] = ()


@dataclass(frozen=True)
class Specialist[Job, Out]:
    """A narrow job two models can do: `ask` puts it to the fine-tune and raises
    `SpecialistUnreachable` if it cannot, `fallback` puts it to the generalist, and `check` turns
    either reply into the answer or raises `Invalid`."""

    name: str
    ask: Callable[[Job], Told]
    fallback: Callable[[Job], Told]
    check: Callable[[Job, Told], Out]


@dataclass(frozen=True)
class FigureJob:
    """What the report analyst is asked: the question as it was put, and the one report to search,
    by ticker and the financial year it ends."""

    question: str
    ticker: str
    year: int


@dataclass(frozen=True)
class Found:
    """The report analyst's answer: its figures, and the passage each is printed in, by id."""

    figures: list[SourcedFigure]
    passages: dict[str, Passage]


_SCALED = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*\*\s*(?:10\s*\*\*\s*(\d+)|1e(\d+)|1(0{3,}))\s*\)?")


def canonical(call: str) -> str:
    """A scaling call in the one form the check reads, `N * 10**k`: `N * 10000000`, `N*1e7` and
    `(N * 10**7)` all become `N * 10**7`. Any other call is returned as it came."""
    found = _SCALED.fullmatch(call.strip())
    if found is None:
        return call
    number, power, exponent, zeros = found.groups()
    return f"{number} * 10**{power or exponent or len(zeros)}"


def check_figures(job: FigureJob, told: Told) -> Found:
    """The report analyst's check, all or nothing: the reply is `SourcedFigures`, each figure's
    passage is one its searches returned, a segment it names is printed there, and its value is
    a number that passage prints, as printed or through one of the reply's calculate calls."""
    figures = parse_sourced(told.text)
    if figures is None:
        raise Invalid("the reply holds no figures")
    read = {passage.id: passage for passage in told.read}
    for figure in figures.figures:
        passage = read.get(figure.chunk_id)
        if passage is None:
            raise Invalid(f"{figure.chunk_id} is not a passage its searches returned")
        if figure.basis not in (None, *BASES) and squeeze(figure.basis) not in squeeze(
            passage.text
        ):
            raise Invalid(f"{figure.chunk_id} does not name the segment {figure.basis!r}")
        label = figure.model_dump(exclude={"chunk_id", "basis"})
        backed = any(
            check_call(Label(**label, call=call), passage.text, CHANGE)
            for call in (None, *(canonical(c) for c in told.calls))
        )
        if not backed:
            raise Invalid(f"{figure.chunk_id} does not back {figure.value} for {figure.metric}")
    cited = {figure.chunk_id: read[figure.chunk_id] for figure in figures.figures}
    return Found(figures.figures, cited)
