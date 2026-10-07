"""Runs only when credentials exist. count_tokens is free, so this costs nothing."""

from __future__ import annotations

import pytest

from aigent.config import MODEL
from aigent.llm.adapters.anthropic import get_client, has_credentials

pytestmark = pytest.mark.skipif(not has_credentials(), reason="no Anthropic credentials configured")


def test_count_tokens_roundtrip() -> None:
    client = get_client()
    result = client.messages.count_tokens(
        model=MODEL, messages=[{"role": "user", "content": "hello, world"}]
    )
    assert result.input_tokens > 0
