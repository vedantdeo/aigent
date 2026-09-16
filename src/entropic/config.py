"""Settings: credentials, model choice, and the ceilings we are willing to spend.

Values only, plus the one function that builds a client. Pricing and enforcement live in `pricing`.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()


# MODEL is what Entropic thinks with; JUDGE_MODEL grades an eval, and is deliberately not MODEL.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"
MODEL: str = os.environ.get("ENTROPIC_MODEL", DEFAULT_MODEL)
JUDGE_MODEL: str = os.environ.get("ENTROPIC_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


# --- Budget ceilings -------------------------------------------------------------------------
# Three scales: one request, one interactive run, one eval over a dataset. `pricing` enforces them.

MAX_USD_PER_REQUEST: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_REQUEST", "0.25"))
MAX_USD_PER_RUN: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_RUN", "1.00"))
MAX_USD_PER_EVAL: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_EVAL", "2.00"))


# --- Call shape ------------------------------------------------------------------------------
# One output cap per call site. The pre-flight guard prices the full cap, so an oversized number
# trips the per-request ceiling for nothing; thinking tokens count against it too.

MAX_TOKENS_FIRST_CALL = 1024
MAX_TOKENS_STREAMING = 4096
MAX_TOKENS_EXTRACT = 2048
MAX_TOKENS_HEADLINE = 128  # a five-field record is ~45 tokens with THINKING_EVAL off
MAX_TOKENS_TOOL_LOOP = 4096
MAX_TOKENS_CHAT = 4096
MAX_TOKENS_JUDGE = 1024

# Off for eval runs: thinking wobbles, and Claude 5 deprecated temperature and top_p.
THINKING_EVAL = False

# How many times the agent may go round before giving up.
MAX_AGENT_TURNS = 8

# How many failing rows an eval report prints before it truncates.
MAX_FAILURES_SHOWN = 10


# --- Retrieval -------------------------------------------------------------------------------
# Week 2's knobs. The embedding model runs locally, so none of these spends anything.

# Its 384 dimensions and 512-token limit are read off the model, not set here.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 64

# Query side only. A fallback: `LocalEmbedder` prefers what the model declares in `prompts`.
# Transcribed from the model card, so re-check it against the card when changing `EMBED_MODEL`.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# In characters, not tokens: the tokenizer belongs to the embedding model. ~1200 chars is ~300
# tokens, leaving room under the model's 512 before it starts truncating.
CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200
CHUNK_OVERLAP_SENTENCES = 1
CHUNK_MAX_CHARS = 2000
CHUNK_MIN_CHARS = 80

# What `by_heading` accepts as a heading in PDF text, where no markup survives to say so.
HEADING_MAX_CHARS = 80
HEADING_MAX_WORDS = 12
HEADING_MIN_CAPITAL_RATIO = 0.6

# How many chunks a retriever returns, and therefore how many an answer can cite.
TOP_K = 5


def has_credentials() -> bool:
    """True if the SDK will find something to authenticate with."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def get_client() -> anthropic.Anthropic:
    """Build the SDK client.

    Set ANTHROPIC_WORKSPACE_ID only for an org-level key; the API refuses one without the header.
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
