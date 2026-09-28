"""Settings: credentials, model choice, and the ceilings we are willing to spend.

Values only, plus the one function that builds a client. Pricing and enforcement live in `pricing`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from aigent.adapters import spec
from aigent.adapters.client import Client
from aigent.messages import Thinking

load_dotenv()


# Which client `llm` talks to; its settings live in `adapters/cfg_<name>.py`, and the three models
# below are that client's, overridable one at a time for a run.
CLIENT: str = os.environ.get("AIGENT_CLIENT", "anthropic")
ACTIVE = spec(CLIENT)

MODEL: str = os.environ.get("AIGENT_MODEL", ACTIVE.model)
JUDGE_MODEL: str = os.environ.get("AIGENT_JUDGE_MODEL", ACTIVE.judge_model)
SMALL_MODEL: str = os.environ.get("AIGENT_SMALL_MODEL", ACTIVE.small_model)


# --- Budget ceilings -------------------------------------------------------------------------
# Six scales: one request, one turn of an agent, one interactive run, one workflow, one agent task
# in Project 2, and one eval over a dataset. `pricing` enforces them.

MAX_USD_PER_REQUEST: float = float(os.environ.get("AIGENT_MAX_USD_PER_REQUEST", "0.25"))
MAX_USD_PER_TURN: float = float(os.environ.get("AIGENT_MAX_USD_PER_TURN", "0.60"))
MAX_USD_PER_RUN: float = float(os.environ.get("AIGENT_MAX_USD_PER_RUN", "1.00"))
MAX_USD_PER_WORKFLOW: float = float(os.environ.get("AIGENT_MAX_USD_PER_WORKFLOW", "0.25"))
MAX_USD_PER_EVAL: float = float(os.environ.get("AIGENT_MAX_USD_PER_EVAL", "2.00"))
MAX_USD_PER_TASK: float = float(os.environ.get("AIGENT_MAX_USD_PER_TASK", "0.50"))
# Assumed per web search: the API adds results mid-call, where the free count cannot see them.
WEB_SEARCH_RESULT_TOKENS = 10_000


# --- Call shape ------------------------------------------------------------------------------
# One output cap per call site. The pre-flight guard prices the full cap, so an oversized number
# trips the per-request ceiling for nothing; thinking tokens count against it too. Side by side
# they show which calls are the expensive ones. A client raises or lowers one of these for itself
# in its own `adapters/cfg_<name>.py`, by key, and inherits every cap it does not name.

DEFAULT_MAX_TOKENS: dict[str, int] = {
    "ANSWER": 256,  # two sentences plus five chunk ids is ~140, with thinking off
    "CHAT": 4096,
    "CRITIQUE": 256,  # evaluator-optimizer
    "DRAFT": 384,
    "EXTRACT": 2048,
    "FACTS": 1024,  # chaining: up to six claims, each with a verbatim quote
    "FIRST_CALL": 1024,
    "HEADLINE": 128,  # a five-field record is ~45 tokens with THINKING_EVAL off
    "JUDGE": 512,  # a verdict is a bool and a sentence or two; Opus reasons at length first
    "NOTE": 384,
    "PLAN": 512,  # orchestrator-workers
    "QUESTION": 512,
    "ROUTE": 128,  # routing: a label, a report and one line of reason
    "ROUTED": 768,
    "SECTION": 384,  # parallelization
    "STREAMING": 4096,
    "SYNTHESIS": 768,
    "TOOL_LOOP": 4096,
    "VOTE": 256,
    "WORKER": 384,
}


def max_tokens(site: str, client: Client | None = None) -> int:
    """The cap for one call site: the client's override, or the shared default.

    The active client unless one is named — as it must be by a call sent to a client chosen at run
    time, which the constants below cannot know.
    """
    settings = ACTIVE if client is None else client
    return settings.max_tokens.get(site, DEFAULT_MAX_TOKENS[site])


# Resolved once, so a call site imports a number as it always did. `tests.test_config` checks that
# these names and the table above stay in step.
MAX_TOKENS_ANSWER = max_tokens("ANSWER")
MAX_TOKENS_CHAT = max_tokens("CHAT")
MAX_TOKENS_CRITIQUE = max_tokens("CRITIQUE")
MAX_TOKENS_DRAFT = max_tokens("DRAFT")
MAX_TOKENS_EXTRACT = max_tokens("EXTRACT")
MAX_TOKENS_FACTS = max_tokens("FACTS")
MAX_TOKENS_FIRST_CALL = max_tokens("FIRST_CALL")
MAX_TOKENS_HEADLINE = max_tokens("HEADLINE")
MAX_TOKENS_JUDGE = max_tokens("JUDGE")
MAX_TOKENS_NOTE = max_tokens("NOTE")
MAX_TOKENS_PLAN = max_tokens("PLAN")
MAX_TOKENS_QUESTION = max_tokens("QUESTION")
MAX_TOKENS_ROUTE = max_tokens("ROUTE")
MAX_TOKENS_ROUTED = max_tokens("ROUTED")
MAX_TOKENS_SECTION = max_tokens("SECTION")
MAX_TOKENS_STREAMING = max_tokens("STREAMING")
MAX_TOKENS_SYNTHESIS = max_tokens("SYNTHESIS")
MAX_TOKENS_TOOL_LOOP = max_tokens("TOOL_LOOP")
MAX_TOKENS_VOTE = max_tokens("VOTE")
MAX_TOKENS_WORKER = max_tokens("WORKER")

# Off for eval runs: thinking wobbles, and Claude 5 deprecated temperature and top_p.
THINKING_EVAL = False

# The same setting as the wire wants it. Built once so two call sites cannot disagree about
# whether thinking is on; turning it on means raising the output cap at every site that sends it.
THINKING_EVAL_PARAM: Thinking = (
    {"type": "enabled", "budget_tokens": 1024} if THINKING_EVAL else {"type": "disabled"}
)

# How many times the agent may go round before giving up.
MAX_AGENT_TURNS = 25  # a backstop against a loop of cheap turns; the budget ends a real one first
MIN_TOKENS_FINAL_ANSWER = 1024  # a tool loop out of room answers only if this much output fits

# How many calls one concurrent batch keeps in flight at once.
MAX_PARALLEL_CALLS = 4

# Workflows split a task into small, fully specified calls: the case thinking adds least to.
THINKING_WORKFLOW_PARAM: Thinking = {"type": "disabled"}

# The most subtasks an orchestrator may hand out, and drafts an evaluator may send back.
MAX_PLAN_TASKS = 4
MAX_REFINE_ROUNDS = 3
MAX_GRAPH_STEPS = 10  # LangGraph's own default is 10,007 supersteps

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


# --- Toy fine-tune ---------------------------------------------------------------------------
# Template headlines that train underhood's toy LoRA: how many, the share kept back to validate on,
# the seed that makes a set repeatable, and the client whose wire the rows are rendered for.
SYNTHETIC_HEADLINES = 200
SYNTHETIC_VALID_FRACTION = 0.1
SYNTHETIC_SEED = 0
SYNTHETIC_CLIENT = "local-1.7b"
