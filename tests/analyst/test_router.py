"""The router against scripted models: which path answers, what a failed check costs, and what an
unreachable specialist does."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from aigent.analyst.router import Router, Routing
from aigent.analyst.specialists import Invalid, Specialist, Told
from aigent.errors import SpecialistUnreachable

GOOD, BAD = "42", "forty-two"


def number(job: str, told: Told) -> int:
    """A check for a toy job: the reply is a whole number."""
    if not told.text.isdigit():
        raise Invalid(f"{told.text!r} is not a number")
    return int(told.text)


def says(text: str, usd: float, log: list[str], who: str) -> Callable[[str], Told]:
    def ask(job: str) -> Told:
        log.append(who)
        return Told(text, usd, input_tokens=100, output_tokens=10)

    return ask


def router(generalist_only: bool = False) -> Router:
    ticks = iter(range(0, 1000, 3))
    return Router(
        price=lambda i, o: (i + o) / 100, generalist_only=generalist_only, clock=lambda: next(ticks)
    )


@pytest.mark.parametrize(
    ("small", "big", "only", "asked", "logged"),
    [
        pytest.param(
            GOOD,
            GOOD,
            False,
            ["specialist"],
            Routing("toy", "specialist", None, 0.01, 0.0, 1.1, 3),
            id="a specialist's answer that passes is the answer",
        ),
        pytest.param(
            BAD,
            GOOD,
            False,
            ["specialist", "generalist"],
            Routing("toy", "generalist", "'forty-two' is not a number", 0.01, 0.5, 1.1, 3),
            id="one that fails goes to the generalist, and both are paid for",
        ),
        pytest.param(
            BAD,
            GOOD,
            True,
            ["generalist"],
            Routing("toy", "generalist", None, 0.0, 0.5, 0.0, 3),
            id="the generalist-only arm never asks the specialist",
        ),
    ],
)
def test_the_router_takes_the_cheap_path_when_its_answer_passes(
    small: str, big: str, only: bool, asked: list[str], logged: Routing
) -> None:
    log: list[str] = []
    toy = Specialist(
        "toy", says(small, 0.01, log, "specialist"), says(big, 0.5, log, "generalist"), number
    )
    route = router(only)

    assert route(toy, "job") == 42
    assert log == asked and route.routed == [logged]


@pytest.mark.parametrize(
    ("only", "failure"),
    [
        pytest.param(
            False,
            "'forty-two' is not a number; the generalist: 'forty-two' is not a number",
            id="after the specialist",
        ),
        pytest.param(True, "the generalist: 'forty-two' is not a number", id="on its own"),
    ],
)
def test_a_generalist_answer_that_fails_too_is_an_error_and_is_logged(
    only: bool, failure: str
) -> None:
    log: list[str] = []
    toy = Specialist(
        "toy", says(BAD, 0.01, log, "specialist"), says(BAD, 0.5, log, "generalist"), number
    )
    route = router(only)

    with pytest.raises(Invalid, match="not a number"):
        route(toy, "job")
    [logged] = route.routed
    assert (logged.path, logged.failure) == (None, failure)
    assert logged.generalist_usd == 0.5


def test_a_specialist_that_cannot_be_reached_fails_the_task_without_a_fallback() -> None:
    log: list[str] = []

    def down(job: str) -> Told:
        raise SpecialistUnreachable("hostllm did not answer in 180 s")

    route = router()

    with pytest.raises(SpecialistUnreachable, match="did not answer"):
        route(Specialist("toy", down, says(GOOD, 0.5, log, "generalist"), number), "job")
    assert log == [], "the generalist is not asked"
    assert route.routed == [
        Routing("toy", None, "unreachable: hostllm did not answer in 180 s", 0.0, 0.0, 0.0, 3)
    ]
