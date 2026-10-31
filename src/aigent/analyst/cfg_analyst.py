"""Settings for the analyst, the capstone: its models, how they think, its loop and cost ceilings,
its run's shape, and its own output caps. Split from `config` by concern; a cap here overrides
`config.DEFAULT_MAX_TOKENS` for the analyst, and a client's own cap overrides both.
"""

from __future__ import annotations

from aigent import config
from aigent.llm.adapters.client import Client
from aigent.llm.messages import Thinking

# The models, and how the generalist thinks.
GENERALIST_MODEL = "claude-sonnet-5"  # the default generalist
GENERALIST_THINKING: Thinking = {"type": "adaptive"}  # pass None for a model that takes none
GENERALIST_CALM: Thinking = {"type": "disabled"}  # a calm generalist's, once it may stop thinking
CRITIC_MODEL = "claude-opus-5"  # never the generalist's own model

# The loops.
MAX_CRITIC_REVISIONS = 2  # redrafts the critic may send back before a human sees the memo
SPECIALIST_TRIES = 2  # a request the endpoint drops is sent once more, then the task fails
SPECIALIST_TOOL_TURNS = 6  # two searches, two calculate turns, the answer, one to spare

# `python -m aigent.analyst.run`: each question's ceilings, its specialists' too, and its tools.
MAX_USD_ANALYST_GENERALIST = 0.40
MAX_USD_ANALYST_CRITIC = 0.30
MAX_USD_ANALYST_SPECIALIST = 0.30
ANALYST_SERVED_TOOLS = ("calculate", "compare_unverified", "fundamentals", "search_filings")
ANALYST_WORKERS = 4  # questions in flight at once, so the specialists' GPU is shared, not idle

# The analyst's own call sites' caps, over the shared table.
MAX_TOKENS: dict[str, int] = {
    "MEMO": 2048,  # a summary, up to a dozen cited claims, caveats
    "VERDICT": 1024,  # a verdict and a quoted sentence, after Opus reasons
    "RESEARCH_TURN": 1024,  # a turn of the report analyst's loop: sourced figures, or tool calls
}


def max_tokens(site: str, client: Client) -> int:
    """The cap for one of the analyst's call sites on `client`: the client's own, else the
    analyst's, else the shared default."""
    if site in client.max_tokens:
        return client.max_tokens[site]
    return MAX_TOKENS.get(site) or config.max_tokens(site, client)
