"""Shared configuration: credentials, model choice, and cost accounting.

Every script in this repo prints what it spent. Get in the habit now; you will be asked about
cost in every interview.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
from anthropic.types import Usage
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


def has_credentials() -> bool:
    """True if the SDK will find something to authenticate with."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def get_client() -> anthropic.Anthropic:
    if not has_credentials():
        raise SystemExit(
            "No Anthropic credentials found.\n"
            "  Option 1: copy .env.example to .env and set ANTHROPIC_API_KEY\n"
            "  Option 2: install the `ant` CLI and run `ant auth login`\n"
        )
    return anthropic.Anthropic()
