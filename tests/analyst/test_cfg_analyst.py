"""The analyst's own caps: a client's cap wins, then the analyst's, then the shared default."""

from __future__ import annotations

import pytest

from aigent import config
from aigent.analyst.cfg_analyst import MAX_TOKENS, max_tokens
from aigent.llm.adapters import spec
from aigent.llm.adapters.cfg_sarvam import THOUGHT


@pytest.mark.parametrize(
    ("site", "client", "expected"),
    [
        pytest.param("MEMO", "anthropic", MAX_TOKENS["MEMO"], id="the analyst's own cap"),
        pytest.param("VERDICT", "anthropic", MAX_TOKENS["VERDICT"], id="the critic's"),
        pytest.param("MEMO", "sarvam-thinking", THOUGHT["MEMO"], id="a client's cap wins"),
        pytest.param(
            "TOOL_LOOP",
            "anthropic",
            config.DEFAULT_MAX_TOKENS["TOOL_LOOP"],
            id="a site the analyst leaves alone: the shared default",
        ),
    ],
)
def test_a_cap_is_the_client_s_then_the_analyst_s_then_the_shared_one(
    site: str, client: str, expected: int
) -> None:
    assert max_tokens(site, spec(client)) == expected


def test_the_analyst_s_caps_are_its_own_not_the_shared_table_s() -> None:
    assert not set(MAX_TOKENS) & set(config.DEFAULT_MAX_TOKENS), "one place per cap"
