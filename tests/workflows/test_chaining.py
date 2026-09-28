"""Prompt chaining: the gate between the two calls is code, so it can be tested as a table.

The load-bearing test is the empty-gate one. A chain that writes its note whether or not any fact
survived has a gate that decides nothing, and pays for a note built from nothing checked.
"""

from __future__ import annotations

import pytest

from aigent.workflows.chaining import Fact, Facts, gate, run

from ..conftest import PASSAGES, FakeSearch, MakeLlm, Sent

EVERYTHING = [chunk for chunks in PASSAGES.values() for chunk in chunks]
REVENUE = PASSAGES["ITC-FY25"][0]
HELD = "net segment revenue grew 6.5 per cent"


@pytest.mark.parametrize(
    ("quote", "passage", "kept"),
    [
        pytest.param(HELD, REVENUE.id, True, id="a verbatim quote from the passage it cites"),
        pytest.param("Net  segment REVENUE grew 6.5", REVENUE.id, True, id="spacing and case"),
        pytest.param("revenue grew 9 per cent", REVENUE.id, False, id="a figure it never stated"),
        pytest.param(HELD, "RELIANCE-FY25#0001", False, id="a real quote cited to the wrong one"),
        pytest.param(HELD, "ITC-FY25#9999", False, id="a passage that was never retrieved"),
        pytest.param("per cent", REVENUE.id, False, id="too short to prove anything"),
    ],
)
def test_a_fact_passes_the_gate_only_if_its_passage_holds_its_quote(
    quote: str, passage: str, kept: bool
) -> None:
    fact = Fact(claim="a claim", quote=quote, passage=passage)

    survivors, dropped = gate([fact], EVERYTHING)

    assert (survivors == [fact]) is kept, (survivors, dropped)
    assert len(survivors) + len(dropped) == 1


def test_when_nothing_survives_the_gate_no_note_is_paid_for(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    invented = Fact(claim="made up", quote="revenue doubled to a record", passage=REVENUE.id)
    llm, fake = make_llm(lambda sent: Facts(facts=[invented]))

    result = run(llm, search, "how did cigarettes do?")

    assert result.note is None and result.dropped == [invented]
    assert [sent.kind for sent in fake.sent] == ["parse"], "the write step must never be sent"


def test_the_note_is_written_from_the_surviving_facts_alone(
    make_llm: MakeLlm, search: FakeSearch
) -> None:
    real = Fact(claim="Cigarette revenue grew 6.5%", quote=HELD, passage=REVENUE.id)
    invented = Fact(claim="Margins tripled", quote="operating margins tripled", passage=REVENUE.id)

    def reply(sent: Sent) -> Facts | str:
        return Facts(facts=[real, invented]) if sent.schema is Facts else "the note"

    llm, fake = make_llm(reply)

    result = run(llm, search, "how did cigarettes do?")

    assert result.note == "the note"
    written = fake.sent[-1].prompt
    assert real.claim in written and invented.claim not in written, written
