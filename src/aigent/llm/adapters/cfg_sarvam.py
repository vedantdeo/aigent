"""Settings for the `sarvam` clients: Sarvam AI's hosted API, over the OpenAI wire; `sarvam`
with reasoning off, `sarvam-thinking` with it on high.
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
)

# Every cap a capstone generalist sends, raised for reasoning, which is billed as output.
THOUGHT = {site: 16384 for site in ("FIGURE_ANSWER", "HEADLINE", "MEMO", "TOOL_LOOP")}

THINKING = replace(
    CLIENT,
    name="sarvam-thinking",
    max_tokens=THOUGHT,
    extra_body={"reasoning_effort": "high"},
)

CLIENTS = {client.name: client for client in (CLIENT, THINKING)}
