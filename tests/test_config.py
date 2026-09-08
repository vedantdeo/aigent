from __future__ import annotations

import pytest

from entropic.config import cost_usd


def test_opus_pricing() -> None:
    # 1M input at $5 and 1M output at $25
    assert cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_cache_multipliers() -> None:
    write = cost_usd("claude-opus-5", 0, 0, cache_write_tokens=1_000_000)
    read = cost_usd("claude-opus-5", 0, 0, cache_read_tokens=1_000_000)
    assert write == pytest.approx(6.25)
    assert read == pytest.approx(0.5)


def test_unknown_model_is_free_not_fatal() -> None:
    assert cost_usd("claude-does-not-exist", 10, 10) == 0.0
