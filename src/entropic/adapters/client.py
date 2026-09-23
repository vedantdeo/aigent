"""The shape of a client's settings. One `cfg_<name>.py` beside this fills it in per client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Client:
    """One endpoint, and everything about it the rest of the repo would otherwise guess.

    `wire` names the adapter that speaks for it, and several clients may share one. `max_tokens`
    overrides the shared per-call-site caps in `config` by their key, and holds only what differs:
    a client happy with every default restates nothing.
    """

    name: str
    wire: str
    model: str
    judge_model: str
    small_model: str
    base_url: str | None = None
    max_tokens: Mapping[str, int] = field(default_factory=dict)
