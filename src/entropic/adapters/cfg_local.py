"""Settings for the `local` client: a model on this machine, behind `mlx_lm.server`.

Started with the model this names, which the server resolves from the Hugging Face cache:

    uv run mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080

One client, not one per model: ENTROPIC_MODEL swaps which one is asked for, and the server loads
it on the first request. Every model named here or swapped in needs a row in `pricing.PRICES`.
"""

from __future__ import annotations

from entropic.adapters.client import Client

# It speaks the OpenAI wire, as vLLM and OpenAI itself do; only the address differs.
CLIENT = Client(
    name="local",
    wire="openai",
    model="mlx-community/Qwen3-4B-Instruct-2507-4bit",
    # A 4B model grading its own answers is not a second opinion, but there is no cheaper local
    # model worth asking: the same weights judge, and the eval says so.
    judge_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    small_model="mlx-community/Qwen3-4B-Instruct-2507-4bit",
    base_url="http://127.0.0.1:8080/v1",
    # It has no constrained decoding, so a record is prompted for and parsed: leave it room for
    # the JSON and a stray line of preamble the parser will strip.
    max_tokens={"ANSWER": 512, "JUDGE": 512},
)
