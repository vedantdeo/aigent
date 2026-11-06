"""The standard formulas a metric is built on: plain arithmetic over named inputs, with no finance
and no period in them, so one formula serves every metric that needs it and is checked once, in a
unit test. It imports nothing from the package, so it can be lifted elsewhere unchanged."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Unit = Literal["pure", "same"]  # a plain number, or the unit its inputs share


@dataclass(frozen=True)
class Formula:
    """A named function of named inputs, and the unit of what it returns."""

    name: str
    variables: tuple[str, ...]
    unit: Unit
    description: str
    compute: Callable[..., float]


def _nonzero(value: float, name: str) -> float:
    if value == 0:
        raise ValueError(f"{name} is zero")
    return value


def growth(current: float, previous: float) -> float:
    """The change from `previous` to `current` as a share of `previous`'s size."""
    return (current - previous) / abs(_nonzero(previous, "previous"))


def ratio(numerator: float, denominator: float) -> float:
    return numerator / _nonzero(denominator, "denominator")


def difference(minuend: float, subtrahend: float) -> float:
    return minuend - subtrahend


def total(first: float, second: float) -> float:
    return first + second


def average(first: float, second: float) -> float:
    return (first + second) / 2


FORMULAS: dict[str, Formula] = {
    f.name: f
    for f in (
        Formula("average", ("first", "second"), "same", "(first + second) / 2", average),
        Formula(
            "difference",
            ("minuend", "subtrahend"),
            "same",
            "minuend - subtrahend",
            difference,
        ),
        Formula(
            "growth",
            ("current", "previous"),
            "pure",
            "(current - previous) / |previous|",
            growth,
        ),
        Formula("ratio", ("numerator", "denominator"), "pure", "numerator / denominator", ratio),
        Formula("total", ("first", "second"), "same", "first + second", total),
    )
}
