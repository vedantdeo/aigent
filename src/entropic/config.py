"""Settings: credentials, model choice, and the ceilings we are willing to spend.

Values only, plus the one function that builds a client. Pricing and enforcement live in `pricing`.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from anthropic.types import ThinkingConfigParam
from dotenv import load_dotenv

load_dotenv()


# MODEL is what Entropic thinks with; JUDGE_MODEL grades an eval, and is deliberately not MODEL.
# SMALL_MODEL is where the routing workflow sends the questions that do not need MODEL.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"
DEFAULT_SMALL_MODEL = "claude-haiku-4-5"
MODEL: str = os.environ.get("ENTROPIC_MODEL", DEFAULT_MODEL)
JUDGE_MODEL: str = os.environ.get("ENTROPIC_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
SMALL_MODEL: str = os.environ.get("ENTROPIC_SMALL_MODEL", DEFAULT_SMALL_MODEL)


# --- Budget ceilings -------------------------------------------------------------------------
# Five scales: one request, one turn of an agent, one interactive run, one workflow, one eval over
# a dataset. `pricing` enforces them.

MAX_USD_PER_REQUEST: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_REQUEST", "0.25"))
MAX_USD_PER_TURN: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_TURN", "0.40"))
MAX_USD_PER_RUN: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_RUN", "1.00"))
MAX_USD_PER_WORKFLOW: float = float(os.environ.get("ENTROPIC_MAX_USD_PER_WORKFLOW", "0.25"))
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
MAX_TOKENS_JUDGE = 256  # a verdict is one or two sentences plus a bool
MAX_TOKENS_QUESTION = 512
MAX_TOKENS_ANSWER = 256  # two sentences plus five chunk ids is ~140, with thinking off

# The workflows' call sites, grouped by pattern.
MAX_TOKENS_FACTS = 1024  # chaining: up to six claims, each with a verbatim quote
MAX_TOKENS_NOTE = 384
MAX_TOKENS_ROUTE = 128  # routing: a label, a report and one line of reason
MAX_TOKENS_ROUTED = 768
MAX_TOKENS_SECTION = 384  # parallelization
MAX_TOKENS_VOTE = 256
MAX_TOKENS_PLAN = 512  # orchestrator-workers
MAX_TOKENS_WORKER = 384
MAX_TOKENS_SYNTHESIS = 768
MAX_TOKENS_DRAFT = 384  # evaluator-optimizer
MAX_TOKENS_CRITIQUE = 256

# Off for eval runs: thinking wobbles, and Claude 5 deprecated temperature and top_p.
THINKING_EVAL = False

# The same setting as the wire wants it. Built once so two call sites cannot disagree about
# whether thinking is on; turning it on means raising the output cap at every site that sends it.
THINKING_EVAL_PARAM: ThinkingConfigParam = (
    {"type": "enabled", "budget_tokens": 1024} if THINKING_EVAL else {"type": "disabled"}
)

# How many times the agent may go round before giving up.
MAX_AGENT_TURNS = 8
MIN_TOKENS_FINAL_ANSWER = 1024  # a tool loop out of room answers only if this much output fits

# How many calls one concurrent batch keeps in flight at once.
MAX_PARALLEL_CALLS = 4

# Workflows split a task into small, fully specified calls: the case thinking adds least to.
THINKING_WORKFLOW_PARAM: ThinkingConfigParam = {"type": "disabled"}

# The most subtasks an orchestrator may hand out, and drafts an evaluator may send back.
MAX_PLAN_TASKS = 4
MAX_REFINE_ROUNDS = 3

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

# BM25's two shape parameters, at the values the literature settled on. `K1` is where term
# frequency saturates; `B` is how hard a long chunk is penalised (0 not at all, 1 fully).
BM25_K1 = 1.5
BM25_B = 0.75

# How deep each ranker's list goes into fusion. Past this the reciprocal contributions are all
# but identical, so the tail adds runtime and noise rather than signal.
FUSE_DEPTH = 100

# Reciprocal rank fusion's damping constant. At 60 the gap between rank 1 and rank 2 is small, so
# a chunk both rankers like beats one that only the loudest ranker put first. Lower it to trust
# first places more.
RRF_K = 60

# The reranker reads every candidate with the query, so this is a forward pass per row, not a
# lookup. Retrieve this many cheaply, score them all, keep the best TOP_K.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 30
RERANK_BATCH = 32

# The method to rank with outside the eval: the best on questions not written from the passages.
SEARCH_METHOD = "hybrid+rerank"

# What counts as page furniture in a PDF, where no markup survives to say what is a running header.
# Only lines this close to the top or bottom of a page are candidates; one whose digits-masked form
# reaches this fraction of the pages is furniture. Below the page floor, nothing is.
FURNITURE_MIN_PAGES = 4
FURNITURE_EDGE_LINES = 2
FURNITURE_RATIO = 0.2

# What joins two pages into one document text. Read as a paragraph break by every splitter.
PAGE_SEPARATOR = "\n\n"


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
