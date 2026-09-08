"""Shared configuration: credentials, model choice, and cost accounting.

Every script in this repo prints what it spent. Get in the habit now; you will be asked about
cost in every interview.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anthropic
from anthropic import Omit, omit
from anthropic.types import MessageParam, ToolParam, Usage
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-opus-5"
MODEL: str = os.environ.get("ENTROPIC_MODEL", DEFAULT_MODEL)


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25

    @property
    def cache_read(self) -> float:
        return self.input * 0.10


PRICES: dict[str, Price] = {
    "claude-opus-5": Price(input=5.0, output=25.0),
    "claude-sonnet-5": Price(input=2.0, output=10.0),
    "claude-haiku-4-5": Price(input=1.0, output=5.0),
    "claude-fable-5-1": Price(input=10.0, output=50.0),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Dollar cost of one request. Unknown models cost nothing rather than crashing the script."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    per_token = 1e-6
    return per_token * (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_write_tokens * price.cache_write
        + cache_read_tokens * price.cache_read
    )


def usage_cost(model: str, usage: Usage) -> float:
    return cost_usd(
        model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
    )


def describe_usage(model: str, usage: Usage) -> str:
    """One line you can paste into LOG.md."""
    return (
        f"[{model}] in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_write={usage.cache_creation_input_tokens or 0} "
        f"cache_read={usage.cache_read_input_tokens or 0} "
        f"cost=${usage_cost(model, usage):.5f}"
    )


# --- Budget guards ---------------------------------------------------------------------------
# Two ceilings in USD, each overridable from .env. A per-eval ceiling arrives with the eval
# harness in Week 2; nothing consumes it before then. A tripped guard is information, not an
# obstacle: the error says what you were about to spend and which knob to turn.

MAX_USD_PER_REQUEST: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_REQUEST", "0.25"))
MAX_USD_PER_RUN: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_RUN", "1.00"))


class BudgetExceeded(RuntimeError):
    """Raised before money is spent (per request) or right after a ceiling is crossed (per run)."""


def worst_case_usd(model: str, input_tokens: int, max_tokens: int) -> float:
    """The most one request can cost: all input at the input price, the full output cap at the
    output price. Thinking counts against max_tokens, so this really is the ceiling."""
    return cost_usd(model, input_tokens=input_tokens, output_tokens=max_tokens)


def assert_request_within_budget(
    model: str, input_tokens: int, max_tokens: int, limit_usd: float = MAX_USD_PER_REQUEST
) -> float:
    """Pure check, no network. Returns the worst-case cost, or raises BudgetExceeded."""
    worst = worst_case_usd(model, input_tokens, max_tokens)
    if worst > limit_usd:
        raise BudgetExceeded(
            f"request could cost up to ${worst:.4f} ({input_tokens} input tokens plus "
            f"max_tokens={max_tokens} on {model}), above the per-request ceiling of "
            f"${limit_usd:.4f}. Trim the input, lower max_tokens, or raise "
            "ENTROPIC_MAX_USD_PER_REQUEST in .env."
        )
    return worst


def check_request(
    client: anthropic.Anthropic,
    *,
    model: str,
    max_tokens: int,
    messages: Sequence[MessageParam],
    system: str | Omit = omit,
    tools: Sequence[ToolParam] | Omit = omit,
) -> int:
    """Count the input tokens (a free call), then enforce the per-request ceiling.

    Pass exactly what the real request will send, so the count is the real count. Returns the
    input token count so callers can print it.
    """
    count = client.messages.count_tokens(model=model, messages=messages, system=system, tools=tools)
    assert_request_within_budget(model, count.input_tokens, max_tokens)
    return count.input_tokens


@dataclass
class Budget:
    """Running spend for one run: a tool loop, a chat session, an eval. Trips after the call that
    crosses the ceiling, so spent_usd always reflects what was actually billed."""

    limit_usd: float = MAX_USD_PER_RUN
    spent_usd: float = 0.0

    def add(self, model: str, usage: Usage) -> float:
        self.spent_usd += usage_cost(model, usage)
        if self.spent_usd > self.limit_usd:
            raise BudgetExceeded(
                f"run has spent ${self.spent_usd:.4f}, above the per-run ceiling of "
                f"${self.limit_usd:.4f}. Raise ENTROPIC_MAX_USD_PER_RUN in .env if this was "
                "intended."
            )
        return self.spent_usd


def has_credentials() -> bool:
    """True if the SDK will find something to authenticate with."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def get_client() -> anthropic.Anthropic:
    """Build the SDK client.

    An organization-level API key is not tied to a workspace, and the API refuses such a key unless
    every request names one via the `anthropic-workspace-id` header. A workspace-scoped key needs no
    header. Set ANTHROPIC_WORKSPACE_ID in .env only if you use an org-level key.
    """
    if not has_credentials():
        raise SystemExit(
            "No Anthropic credentials found.\n"
            "  Option 1: copy .env.example to .env and set ANTHROPIC_API_KEY\n"
            "  Option 2: install the `ant` CLI and run `ant auth login`\n"
        )
    headers: dict[str, str] = {}
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        headers["anthropic-workspace-id"] = workspace_id
    return anthropic.Anthropic(default_headers=headers)
