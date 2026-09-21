"""What every pattern test searches: one passage per report. The `Llm` comes from `make_llm`."""

from __future__ import annotations

import pytest

from entropic.retrieval.chunk import Chunk


def _passage(doc_id: str, text: str) -> Chunk:
    return Chunk(id=f"{doc_id}#0001", text=text, doc_id=doc_id, ordinal=1, start=0, page=12)


PASSAGES: dict[str, list[Chunk]] = {
    "ITC-FY25": [
        _passage("ITC-FY25", "Cigarettes net segment revenue grew 6.5 per cent during the year.")
    ],
    "RELIANCE-FY25": [
        _passage("RELIANCE-FY25", "The Board recommended a dividend of Rs 5.50 per equity share.")
    ],
    "TATAMOTORS-FY25": [
        _passage("TATAMOTORS-FY25", "Jaguar Land Rover delivered record free cash flow in FY25.")
    ],
}


class FakeSearch:
    """Every report's passages when unscoped, one report's when scoped; every query logged."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str | None]] = []

    def __call__(self, query: str, /, *, doc_id: str | None = None) -> list[Chunk]:
        self.asked.append((query, doc_id))
        if doc_id is None:
            return [chunk for chunks in PASSAGES.values() for chunk in chunks]
        return list(PASSAGES[doc_id])


@pytest.fixture
def search() -> FakeSearch:
    return FakeSearch()
