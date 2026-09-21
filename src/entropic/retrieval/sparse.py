"""The sparse index: BM25 over the same chunks the dense index holds.

The lexical half of hybrid retrieval, and the half that finds what an embedding smears. A dense
model encodes what a passage is *about*, so `SACE`, `Rs 34,000 crores` and `KPMG` — rare tokens
carrying the whole answer — sit close to their neighbours in vector space. BM25 scores a chunk on
the query terms it literally contains, weighted by how rare they are: the opposite failure mode,
and the reason fusing the two is worth anything.

Written out rather than imported: it is sixty lines, it keeps the dependency list honest, and
`idf`, `k1` and `b` are easier to reason about when they are in front of you.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from entropic.config import BM25_B, BM25_K1, TOP_K
from entropic.retrieval.chunk import Inventory
from entropic.retrieval.hits import Hit

# Words and bare numbers. `34,000` and `52.3` survive as one token, which is the point of having a
# lexical ranker at all — splitting them hands the digits back to the same blur BM25 exists to fix.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenise(text: str) -> list[str]:
    """Lowercased words and numbers. No stemming and no stopword list.

    `idf` already discounts a term appearing everywhere, which is what a stopword list
    approximates, so the simpler thing is also the more honest one.
    """
    return _TOKEN.findall(text.casefold())


@dataclass(frozen=True)
class SparseIndex:
    """An inverted index over an `Inventory`, plus what scoring needs to know about the corpus.

    `postings` maps a term to the rows holding it and how often, so scoring a query touches only
    the chunks that share a term with it rather than all of them. Ids are the inventory's ids, so
    a hit here is comparable with a `DenseIndex` hit over the same inventory — which is what lets
    the two be fused at all (invariant 17).
    """

    ids: tuple[str, ...]
    postings: dict[str, list[tuple[int, int]]]
    lengths: tuple[int, ...]
    idf: dict[str, float] = field(default_factory=dict)
    average_length: float = 0.0
    k1: float = BM25_K1
    b: float = BM25_B

    @classmethod
    def build(cls, inventory: Inventory, *, k1: float = BM25_K1, b: float = BM25_B) -> SparseIndex:
        """Tokenise every chunk once, invert it, and precompute each term's inverse frequency."""
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for row, text in enumerate(inventory.texts()):
            counts = Counter(tokenise(text))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((row, frequency))

        documents = len(lengths)
        # Robertson/Sparck Jones idf, with the +1 that keeps a term present in every chunk at a
        # small positive weight rather than a negative one — which would score its absence as
        # evidence and rank a chunk higher for *not* containing a query word.
        idf = {
            term: math.log(1 + (documents - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in postings.items()
        }
        return cls(
            ids=tuple(inventory.ids()),
            postings=dict(postings),
            lengths=tuple(lengths),
            idf=idf,
            average_length=(sum(lengths) / documents) if documents else 0.0,
            k1=k1,
            b=b,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def score(self, query: str) -> dict[int, float]:
        """Row → score, for the rows sharing at least one term with the query.

        Sparse on purpose: a chunk sharing no term is absent rather than zero, so `search` can tell
        "no match" from "a match that scored badly" without a threshold.
        """
        scores: dict[int, float] = defaultdict(float)
        if not self.average_length:
            return {}
        for term in tokenise(query):
            weight = self.idf.get(term)
            if weight is None:
                continue
            for row, frequency in self.postings[term]:
                # Saturating term frequency: the tenth `dividend` adds far less than the second,
                # and `b` decides how much of a long chunk's extra hits are discounted as length.
                norm = 1 - self.b + self.b * self.lengths[row] / self.average_length
                scores[row] += weight * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
        return dict(scores)

    def search(self, query: str, k: int = TOP_K) -> list[Hit]:
        """The k best-scoring chunks, best first, ties breaking by position as the dense index does.

        A chunk sharing no term with the query is left out rather than padded in to reach k: it is
        not a weak match, it is not a match, and passing it on would put text this ranker never
        endorsed in front of a reader.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        scores = self.score(query)
        ranked = sorted(scores, key=lambda row: (-scores[row], row))
        return [Hit(self.ids[row], scores[row]) for row in ranked[:k]]

    def rank_all(self, query: str) -> list[Hit]:
        """Every chunk sharing a term with the query, best first — the input fusion wants.

        Fusion reads ranks, so it needs the whole ranking rather than its top slice: a chunk at
        rank 8 here and rank 3 in the dense ranking should win, and it cannot if this list stopped
        at 5.
        """
        return self.search(query, k=max(len(self.ids), 1))
