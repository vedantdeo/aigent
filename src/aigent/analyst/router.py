"""The router: a specialist's answer when it passes its check, the generalist's when it does not,
and every call logged. A specialist that cannot be reached fails the task; nothing answers in its
place.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from aigent.analyst.specialists import Invalid, Specialist
from aigent.errors import SpecialistUnreachable

Path = Literal["specialist", "generalist"]


@dataclass(frozen=True)
class Routing:
    """One routed call: the path whose answer was used (None when neither passed), why the
    specialist's was not, what each path cost, and what the specialist's tokens would have cost at
    the generalist's prices."""

    specialist: str
    path: Path | None
    failure: str | None
    specialist_usd: float
    generalist_usd: float
    counterfactual_usd: float
    seconds: float


@dataclass
class Router:
    """Runs each specialist call and logs it in `routed`. `price` is the generalist's cost for
    (input, output) tokens; `generalist_only` is the eval's other arm, skipping the specialist."""

    price: Callable[[int, int], float]
    generalist_only: bool = False
    clock: Callable[[], float] = time.monotonic
    routed: list[Routing] = field(default_factory=list[Routing])

    def __call__[Job, Out](self, specialist: Specialist[Job, Out], job: Job) -> Out:
        began = self.clock()
        failure: str | None = None
        small_usd = would_cost = 0.0

        def log(path: Path | None, why: str | None, big_usd: float = 0.0) -> None:
            took = self.clock() - began
            self.routed.append(
                Routing(specialist.name, path, why, small_usd, big_usd, would_cost, took)
            )

        if not self.generalist_only:
            try:
                told = specialist.ask(job)
            except SpecialistUnreachable as down:
                log(None, f"unreachable: {down}")
                raise
            small_usd = told.usd
            would_cost = self.price(told.input_tokens, told.output_tokens)
            try:
                answer = specialist.check(job, told)
            except Invalid as refused:
                failure = str(refused)
            else:
                log("specialist", None)
                return answer
        told = specialist.fallback(job)
        try:
            answer = specialist.check(job, told)
        except Invalid as refused:
            both = f"the generalist: {refused}"
            log(None, both if failure is None else f"{failure}; {both}", told.usd)
            raise
        log("generalist", failure, told.usd)
        return answer
