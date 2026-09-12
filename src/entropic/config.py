"""Settings: credentials, model choice, and the ceilings we are willing to spend.

Values only, plus the one function that turns them into a client. What a call *costs*, and whether
it is allowed, lives in `pricing` — this module knows nothing about money beyond the three numbers
below.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()


# Two defaults, because the two jobs are not the same job.
#
# MODEL is what Entropic thinks with: the CLI, the tool loop, the primitives. Pick for capability.
#
# JUDGE_MODEL is what grades an eval. Deliberately a *different* model, for two reasons. A model
# asked to grade its own output favours it — using another model removes that by construction
# instead of by remembering to. And applying a written rubric to a short answer is a far easier task
# than producing the answer, so paying Opus rates per row to do it is waste: a judge call happens
# once per case, which doubles the cost of an eval if you let it.
#
# Point both at the same model and you are letting it mark its own homework. Sometimes that is fine
# — a format check, a yes/no with no room to flatter itself — but make it a choice.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"
MODEL: str = os.environ.get("ENTROPIC_MODEL", DEFAULT_MODEL)
JUDGE_MODEL: str = os.environ.get("ENTROPIC_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


# --- Budget ceilings -------------------------------------------------------------------------
# Three ceilings in USD, each overridable from .env. They sit at different scales: one request, one
# interactive run, one eval over a whole dataset. The numbers live here with the rest of the
# settings; `pricing` is what enforces them. A tripped guard is information, not an obstacle: the
# message says what you were about to spend and which knob to turn.

MAX_USD_PER_REQUEST: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_REQUEST", "0.25"))
MAX_USD_PER_RUN: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_RUN", "1.00"))
MAX_USD_PER_EVAL: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_EVAL", "2.00"))


# --- Call shape ------------------------------------------------------------------------------
# Output caps, one per call site. They are here rather than next to each call so the whole set is
# visible at once — side by side you can see that the agent loop and chat are the expensive ones and
# a first call is not, which is invisible when each number sits alone in its own module.
#
# max_tokens is a ceiling, not a target: you are billed for what comes back, but the pre-flight
# guard prices the full cap, so a number set far above what a call needs will trip the per-request
# ceiling for no reason. Thinking tokens count against it too.

MAX_TOKENS_FIRST_CALL = 1024
MAX_TOKENS_STREAMING = 4096
MAX_TOKENS_EXTRACT = 2048
MAX_TOKENS_HEADLINE = 128  # a five-field record is ~45 tokens with THINKING_EVAL off
MAX_TOKENS_TOOL_LOOP = 4096
MAX_TOKENS_CHAT = 4096
MAX_TOKENS_JUDGE = 1024

# Extended thinking, for runs that are a measurement rather than a conversation. On by default; it
# is billed as output, counts against max_tokens, and wobbles — three identical calls returned 192,
# 88 and 203 output tokens and two different records, and it clipped two rows of a 50-row eval by
# eating the budget before the answer. Claude 5 deprecated temperature and top_p, so this is the
# only determinism knob the API still offers.
THINKING_EVAL = False

# How many times the agent may go round before giving up. An uncapped loop is a cost bug waiting to
# happen; the two dollar ceilings back this up rather than replace it.
MAX_AGENT_TURNS = 8

# How many failing rows an eval report prints before it truncates. Enough to see a pattern, not so
# many that the table scrolls off.
MAX_FAILURES_SHOWN = 10


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
