"""Settings for the specialists hostllm serves: two LoRA fine-tunes of Qwen3-4B on one Modal
endpoint that scales to zero, over the OpenAI wire. A client per adapter, since a request names
its adapter as its model.

The endpoint sits behind Modal's proxy auth, which wants two headers before a GPU starts, and
hostllm's own bearer key. All three come from the environment.
"""

from __future__ import annotations

from dataclasses import replace

from aigent.llm.adapters.client import Client

BASE_URL = "https://vedantdeo1--hostllm-ours.us-east.modal.run/v1"
BASE_MODEL = "Qwen/Qwen3-4B"  # whose tokenizer counts for both adapters
PROXY_AUTH = {"Modal-Key": "HOSTLLM_MODAL_KEY", "Modal-Secret": "HOSTLLM_MODAL_SECRET"}

# A request may find the endpoint cold and has to outlast its start: 34 to 110 s when measured.
COLD_START_TIMEOUT = 180.0

HEADLINE = Client(
    name="hostllm-headline",
    wire="openai",
    model="headline-flavour",
    judge_model="claude-sonnet-5",
    small_model="headline-flavour",
    base_url=BASE_URL,
    timeout=COLD_START_TIMEOUT,
    judge_client="anthropic",
    api_key_env="HOSTLLM_API_KEY",
    headers_env=PROXY_AUTH,
    tokenizer=BASE_MODEL,
    # hostllm always templates with thinking off; this makes a count template the same way.
    template_kwargs={"enable_thinking": False},
)
ANALYST = replace(
    HEADLINE, name="hostllm-analyst", model="report-analyst", small_model="report-analyst"
)

# Client name → settings, in one canonical order.
CLIENTS = {client.name: client for client in (ANALYST, HEADLINE)}
