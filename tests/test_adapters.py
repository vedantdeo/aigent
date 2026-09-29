"""Picking a client by name, and the adapter that speaks its wire."""

from __future__ import annotations

from typing import cast

import httpx2
import openai
import pytest

from aigent.adapters import CLIENTS, build, judge_of, spec
from aigent.config import CLIENT
from aigent.llm import Llm


def test_every_client_builds_an_adapter_that_can_send() -> None:
    """A client maps to exactly one adapter, and several clients may share one — so an adapter is
    not asked to answer to the client's name, only to support something."""
    for name in CLIENTS:
        assert build(name).supports, f"{name}'s adapter supports nothing, so it cannot send"


def test_a_judge_client_is_a_real_client_whose_adapter_differs_from_the_answerers() -> None:
    for client in CLIENTS.values():
        if client.judge_client is not None:
            assert client.judge_client in CLIENTS, client.name
            assert client.judge_client != client.name, client.name


@pytest.mark.parametrize(
    ("client", "judge", "wire"),
    [
        pytest.param("sarvam", "anthropic", "anthropic", id="a client's settings name its judge"),
        pytest.param("local", "local", "openai", id="a client naming none judges itself"),
    ],
)
def test_a_clients_judge_is_a_client_with_its_own_adapter(
    client: str, judge: str, wire: str
) -> None:
    assert judge_of(client) == judge
    assert build(judge_of(client)).name == wire


def test_an_unknown_client_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="no client named 'gpt-9'"):
        build("gpt-9")


def test_an_llm_is_built_from_a_client_name_and_never_names_an_adapter() -> None:
    """The wire is infrastructure. The vendor client behind it is built lazily, so constructing an
    `Llm` needs no credentials — only sending does."""
    assert Llm().client == CLIENT
    assert Llm("anthropic").client == "anthropic"


def test_an_llm_hands_its_vendor_client_on_so_two_can_share_one_connection() -> None:
    scripted = object()
    llm = Llm(sdk=scripted)
    assert llm.sdk is scripted
    assert Llm.for_eval(llm.client, sdk=llm.sdk).sdk is scripted


def test_a_client_hands_its_timeout_and_connection_reuse_to_the_sdk() -> None:
    """The SDKs default to a 600-second read and minutes of connection reuse, which together let
    one dead socket stall an overnight eval for three hours. Both are the client's to set: a
    hosted API answers in seconds, a local server may be loading weights inside the first request.
    """
    # The protocol types `client` as `object`, since what a vendor client *is* differs per wire.
    sdk = cast(openai.OpenAI, build("local").client)
    assert sdk.timeout == spec("local").timeout
    # Typed as the base transport; the pool lives on the concrete one.
    transport = cast(httpx2.HTTPTransport, sdk._client._transport)  # noqa: SLF001
    pool = transport._pool  # noqa: SLF001 - no public accessor for the pool
    assert pool._keepalive_expiry == spec("local").keepalive_seconds  # noqa: SLF001
    assert spec("local").timeout > spec("anthropic").timeout, "a local load is slower than an API"


def test_a_reply_block_carries_no_unset_fields_to_echo_back() -> None:
    """The API sends `caller: null` on a server tool call, and refuses it when echoed back."""
    from anthropic.types import Message, ServerToolUseBlock, Usage

    from aigent.adapters.anthropic import reply_of

    message = Message(
        id="m",
        type="message",
        role="assistant",
        model="claude-sonnet-5",
        content=[ServerToolUseBlock(type="server_tool_use", id="w1", name="web_search", input={})],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    [block] = reply_of(message).blocks
    assert None not in block.values(), block
