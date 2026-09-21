"""The dense index: brute-force cosine search over a matrix of chunk vectors.

No approximate index and no database: a dot product over NumPy is exact and fast enough for this
corpus, so a miss belongs to the chunking or the embedding rather than to an index's own recall
knob. The index holds ids and vectors; `Inventory` holds the chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from entropic.config import TOP_K
from entropic.retrieval.chunk import Inventory
from entropic.retrieval.embed import Embedder, Vectors
from entropic.retrieval.hits import Hit


@dataclass(frozen=True)
class DenseIndex:
    """Chunk ids and their vectors, aligned row for row.

    Rows are L2-normalised, so cosine similarity is `matrix @ query`.
    """

    ids: tuple[str, ...]
    matrix: Vectors
    embedder_name: str = ""

    @classmethod
    def build(cls, inventory: Inventory, embedder: Embedder) -> DenseIndex:
        """Embed every chunk, ids and rows aligned.

        Refuses a vector count that does not match the inventory, and un-normalised rows — both
        would leave every score plausible and every citation wrong.
        """
        vectors = embedder.embed_documents(inventory.texts())
        if len(vectors) != len(inventory):
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(inventory)} chunks"
            )
        if len(vectors) and not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
            raise ValueError("embedder returned un-normalised vectors; cosine would be wrong")
        return cls(
            ids=tuple(inventory.ids()),
            matrix=np.ascontiguousarray(vectors, dtype=np.float32),
            embedder_name=embedder.name,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: Vectors, k: int = TOP_K) -> list[Hit]:
        """The k chunks most similar to a query vector, best first, scored by cosine.

        Ties break by position in the inventory, so two identical runs cannot disagree.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not self.ids:
            return []
        scores = self.matrix @ np.asarray(query, dtype=np.float32)
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.ids[int(row)], float(scores[int(row)])) for row in top]
