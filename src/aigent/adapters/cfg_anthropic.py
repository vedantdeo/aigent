"""Settings for the `anthropic` client: the hosted API, over the Anthropic wire.

Split from `config` by concern and placed beside the adapters: `config` holds the knobs that are
the same whoever answers — spending ceilings, loop caps, the output a task needs — and this holds
what changes with the endpoint.
"""

from __future__ import annotations

from aigent.adapters.client import Client

# MODEL is what aigent thinks with; judge_model grades an eval, and is deliberately not model.
# small_model is where the routing workflow sends the questions that do not need model.
CLIENT = Client(
    name="anthropic",
    wire="anthropic",
    model="claude-opus-5",
    judge_model="claude-sonnet-5",
    small_model="claude-haiku-4-5",
)
