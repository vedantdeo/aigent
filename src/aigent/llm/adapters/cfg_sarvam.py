"""Settings for the `sarvam` clients: Sarvam AI's hosted API, over the OpenAI wire; `sarvam`
with reasoning off, `sarvam-thinking` with it at medium.
"""

from __future__ import annotations

from dataclasses import replace

from aigent.llm.adapters.client import Client

# Room for an answer if reasoning ever comes back on; reasoning is billed as output.
REASONED = {"ANSWER": 2048, "HEADLINE": 2048}

# Sarvam serves one current model, so its answers are graded on the `anthropic` client.
CLIENT = Client(
    name="sarvam",
    wire="openai",
    model="sarvam-105b",
    judge_model="claude-sonnet-5",
    small_model="sarvam-105b",
    base_url="https://api.sarvam.ai/v1",
    judge_client="anthropic",
    api_key_env="SARVAM_API_KEY",
    tokenizer="sarvamai/sarvam-105b",
    max_tokens=REASONED,
    constrains_schema=True,
    # An explicit null turns reasoning off; omitting it means "medium", and "low" filled 2048.
    extra_body={"reasoning_effort": None},
    # A rate limit outlasted rather than failed: up to six more tries, waits growing to a minute.
    retries=6,
    retry_base_seconds=2.0,
    retry_max_seconds=60.0,
)

# Every cap a capstone generalist sends, raised for reasoning, which is billed as output.
THOUGHT = {site: 16384 for site in ("FIGURE_ANSWER", "HEADLINE", "MEMO", "TOOL_LOOP")}

THINKING = replace(
    CLIENT,
    name="sarvam-thinking",
    max_tokens=THOUGHT,
    extra_body={},  # no reasoning_effort: Sarvam reasons at medium, which its enum does not list
)

CLIENTS = {client.name: client for client in (CLIENT, THINKING)}
