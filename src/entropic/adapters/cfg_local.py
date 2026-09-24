"""Settings for the models served on this machine, behind one `mlx_lm.server`.

Start it once; it serves whichever model a request names, from the Hugging Face cache:

    uv run mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080

**One model is resident at a time.** The server swaps when a request names a different one, so a
comparison runs one client's prompts together rather than interleaving them, and the first call
after a switch pays for the load.

`local` is the default. The rest are the same 4B at four bit widths, plus an 8B at 4-bit — the
size-against-precision trade this list exists to measure, within one model family so a comparison
moves one thing. A 14B fits in 16 GB but was left out: these runs are minutes of sustained decode
and the machine gets hot. Every model here needs a row in `pricing.PRICES`.
"""

from __future__ import annotations

from entropic.adapters.client import Client

BASE_URL = "http://127.0.0.1:8080/v1"

# A 4B judging its own answers is not a second opinion, so a bigger model grades. It is a Qwen3
# rather than the Qwen2.5-7B this used to be: a verdict is structured output, and that model lost
# 16 of 54 rows to prose where an object was asked for. Grading is the job it was worst at.
JUDGE = "mlx-community/Qwen3-8B-4bit"

# No constrained decoding here: a record is prompted for and parsed, so leave room for the JSON
# and for the stray line of preamble the parser strips.
PROMPTED_RECORD = {"ANSWER": 512, "JUDGE": 512}


# Qwen3's own models think before answering unless told not to, and thought is billed against the
# output cap — a record would never be reached. The 2507 instruct models ignore this and need it
# not be sent, so it is set per client rather than globally.
NO_THINKING: dict[str, object] = {"enable_thinking": False}


def served(
    name: str, model: str, *, judge: str = JUDGE, template: dict[str, object] | None = NO_THINKING
) -> Client:
    """One model on the local server. These differ only in which weights answer.

    Every client carries `NO_THINKING`, including those whose own model ignores it: the judge is
    reached through the *task's* client, so a client serving a non-thinking model still has to
    tell a thinking judge not to.
    """
    return Client(
        name=name,
        wire="openai",
        model=model,
        judge_model=judge,
        small_model=model,
        base_url=BASE_URL,
        # The first request after a model switch waits for the weights to load, and for a model
        # not yet on this machine, to download.
        timeout=1800.0,
        max_tokens=PROMPTED_RECORD,
        template_kwargs=template or {},
    )


QWEN3_4B_4BIT = served("local", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
QWEN3_4B_6BIT = served("local-4b-6bit", "mlx-community/Qwen3-4B-Instruct-2507-6bit")
QWEN3_4B_8BIT = served("local-4b-8bit", "mlx-community/Qwen3-4B-Instruct-2507-8bit")
QWEN3_4B_BF16 = served("local-4b-bf16", "mlx-community/Qwen3-4B-Instruct-2507-bf16")
# Bigger, and the same family as the 4B above, so a comparison moves size alone. The 8B grades
# the others, and the 4B grades it back — nothing here marks its own paper.
QWEN3_8B_4BIT = served("local-8b-4bit", "mlx-community/Qwen3-8B-4bit", judge=QWEN3_4B_4BIT.model)

CLIENT = QWEN3_4B_4BIT

# Client name → settings, in one canonical order.
CLIENTS = {
    client.name: client
    for client in (
        QWEN3_4B_4BIT,
        QWEN3_4B_6BIT,
        QWEN3_4B_8BIT,
        QWEN3_4B_BF16,
        QWEN3_8B_4BIT,
    )
}
