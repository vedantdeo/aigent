"""`build_search` over fakes for the corpus, the indexes and the ranker: what it returns, and that
concurrent callers never rank at once."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from aigent.retrieval.chunk import Chunk
from aigent.retrieval.hits import Hit
from aigent.workflows import reports

from ..conftest import PASSAGES

CHUNKS = {chunk.id: chunk for chunks in PASSAGES.values() for chunk in chunks}


@dataclass
class _Inventory:
    by_id: dict[str, Chunk] = field(default_factory=lambda: dict(CHUNKS))


@dataclass
class _Ranker:
    """Ranks every chunk, slowly enough that two callers would overlap, and counts the overlap."""

    inside: int = 0
    most_inside: int = 0
    asked: list[tuple[str, str | None]] = field(default_factory=list[tuple[str, str | None]])

    def rank(self, query: str, doc_id: str | None = None) -> list[Hit]:
        self.inside += 1
        self.most_inside = max(self.most_inside, self.inside)
        time.sleep(0.005)
        self.asked.append((query, doc_id))
        self.inside -= 1
        ids = [id_ for id_ in CHUNKS if doc_id is None or id_.startswith(doc_id)]
        return [Hit(id_, 1.0) for id_ in ids]


@pytest.fixture
def ranker(monkeypatch: pytest.MonkeyPatch) -> _Ranker:
    built = _Ranker()
    monkeypatch.setattr(reports, "load_corpus", lambda cache: [])
    monkeypatch.setattr(reports.Inventory, "build", lambda *args: _Inventory())
    monkeypatch.setattr(reports.Indexes, "build", lambda *args, **kwargs: None)
    monkeypatch.setattr(reports, "build_ranker", lambda method, indexes: built)
    return built


def test_search_returns_the_ranked_passages_scoped_as_asked(ranker: _Ranker) -> None:
    search = reports.build_search(cache=False)
    doc_id = next(iter(PASSAGES))

    assert search("dividend") == list(CHUNKS.values())
    assert search("dividend", doc_id=doc_id) == PASSAGES[doc_id]
    assert ranker.asked == [("dividend", None), ("dividend", doc_id)]


def test_concurrent_searches_rank_one_at_a_time(ranker: _Ranker) -> None:
    search = reports.build_search(cache=False)
    callers = [threading.Thread(target=search, args=(f"q{i}",)) for i in range(4)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join()

    assert len(ranker.asked) == 4
    assert ranker.most_inside == 1, f"{ranker.most_inside} searches ranked at once"
