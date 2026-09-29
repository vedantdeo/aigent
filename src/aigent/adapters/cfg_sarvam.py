"""Settings for the `sarvam` client: Sarvam AI's hosted API, over the OpenAI wire."""

from __future__ import annotations

from aigent.adapters.client import Client

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
)
