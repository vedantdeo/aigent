"""Picking a client by name, and the adapter that speaks its wire."""

from __future__ import annotations

import pytest

from entropic.adapters import CLIENTS, build
from entropic.config import CLIENT
from entropic.llm import Llm


def test_every_client_builds_an_adapter_that_can_send() -> None:
    """A client maps to exactly one adapter, and several clients may share one — so an adapter is
    not asked to answer to the client's name, only to support something."""
    for name in CLIENTS:
        assert build(name).supports, f"{name}'s adapter supports nothing, so it cannot send"


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
