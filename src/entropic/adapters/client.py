"""The shape of a client's settings. One `cfg_<name>.py` beside this fills it in per client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from entropic.llm import Request


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
    # How long one request may take before it is abandoned. The SDKs default to a 600-second read
    # with retries, so one stalled call costs half an hour and looks like nothing at all — which
    # is what it did on 2026-09-24, stalling an eval for three hours on a socket with no traffic.
    # A hosted API should answer in seconds; a local server may be loading or fetching weights
    # inside the first request, which is minutes.
    timeout: float = 180.0
    # How long an idle connection may be reused before it is dropped and replaced. The default is
    # minutes, and a call every few seconds keeps one alive far longer than a NAT or firewall on
    # the path keeps its own state: the socket looks open, the request goes into a black hole and
    # the read blocks. That is what stalled an eval for three hours on 2026-09-24.
    keepalive_seconds: float = 30.0
    max_tokens: Mapping[str, int] = field(default_factory=dict)
    # Arguments for the served model's chat template, sent with every request and used when
    # counting, so both sides template alike. `{"enable_thinking": False}` is why this exists:
    # a hybrid-thinking model spends its output cap on thought before it writes any answer.
    template_kwargs: Mapping[str, object] = field(default_factory=dict)


def model_of(request: Request, settings: Client) -> str:
    """The model a request names, or — the usual case — the one its client serves."""
    return request.model or settings.model
