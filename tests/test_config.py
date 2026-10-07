"""The output-cap table, the names derived from it, and a client's right to override one."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aigent import config
from aigent.llm.adapters.client import Client


def test_the_cap_table_and_the_constants_stay_in_step() -> None:
    """Each name is written twice — once as a key, once as a constant — so a test holds them
    together rather than a convention nobody rereads."""
    named = {
        name.removeprefix("MAX_TOKENS_") for name in vars(config) if name.startswith("MAX_TOKENS_")
    }
    assert named == set(config.DEFAULT_MAX_TOKENS)


def test_a_client_overrides_the_caps_it_names_and_inherits_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = Client(
        name="local",
        wire="openai",
        model="qwen3-4b",
        judge_model="qwen3-4b",
        small_model="qwen3-4b",
        max_tokens={"ANSWER": 512},
    )
    monkeypatch.setattr(config, "ACTIVE", local)

    assert config.max_tokens("ANSWER") == 512, "the client's own number wins"
    assert config.max_tokens("JUDGE") == config.DEFAULT_MAX_TOKENS["JUDGE"], "the rest are shared"


def test_a_client_named_at_the_call_site_outranks_the_active_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run can pick its client after import, when every `MAX_TOKENS_*` already names the
    configured one, so a call sent elsewhere asks for its own client's cap."""
    plain = Client(name="plain", wire="anthropic", model="m", judge_model="m", small_model="m")
    local = replace(plain, name="local", wire="openai", max_tokens={"ANSWER": 512})
    monkeypatch.setattr(config, "ACTIVE", plain)

    assert config.max_tokens("ANSWER", local) == 512, "the named client's own number wins"
    assert config.max_tokens("ANSWER") == config.DEFAULT_MAX_TOKENS["ANSWER"], "unnamed: active"


def test_a_cap_no_client_and_no_default_names_is_a_mistake_not_a_zero() -> None:
    with pytest.raises(KeyError):
        config.max_tokens("NO_SUCH_CALL_SITE")


def test_the_cache_defaults_to_five_minutes_sent_as_it_always_was() -> None:
    """The 1-hour cache waits for datasets big enough to outlive five minutes."""
    assert config.CACHE_TTL == "5m"
    assert config.CACHE_PARAM == {"type": "ephemeral"}, "no ttl on the wire for the default"
