"""The dense index: brute-force cosine search over a matrix of chunk vectors.

No approximate index and no database: a dot product over NumPy is exact and fast enough for this
corpus, so a miss belongs to the chunking or the embedding rather than to an index's own recall
knob. The index holds ids and vectors; `Inventory` holds the chunks. `build_cached` keeps the
vectors on disk, keyed by everything that produced them.
"""

from __future__ import annotations

import hashlib
import inspect
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from entropic.config import TOP_K
from entropic.retrieval.chunk import Inventory
from entropic.retrieval.corpus import CACHE_DIR
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
        return cls._checked(tuple(inventory.ids()), vectors, embedder.name)

    @classmethod
    def build_cached(
        cls, inventory: Inventory, embedder: Embedder, cache_dir: Path = CACHE_DIR
    ) -> DenseIndex:
        """`build`, memoised on disk. A damaged or mismatched entry is rebuilt, never raised on."""
        cached = cache_file(inventory, embedder, cache_dir)
        if cached.exists():
            try:
                index = cls.load(cached)
            except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile):
                pass
            else:
                if index.ids == tuple(inventory.ids()) and index.embedder_name == embedder.name:
                    return index

        index = cls.build(inventory, embedder)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for stale in cache_dir.glob(f"index-{inventory.name}-*.npz"):
            stale.unlink(missing_ok=True)
        index.save(cached)
        return index

    def save(self, path: Path) -> None:
        """Write ids, vectors and the embedder's name to `path`, whole or not at all."""
        partial = path.with_suffix(".partial")
        # Handed a file rather than a path, since `np.savez` adds `.npz` to a path without one.
        with partial.open("wb") as handle:
            np.savez(
                handle,
                ids=np.array(self.ids, dtype=np.str_),
                matrix=self.matrix,
                embedder=np.array(self.embedder_name),
            )
        partial.replace(path)

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        """An index `save` wrote, held to the same checks as one just built."""
        with np.load(path, allow_pickle=False) as stored:
            ids = tuple(str(chunk_id) for chunk_id in stored["ids"])
            matrix = np.asarray(stored["matrix"], dtype=np.float32)
            embedder_name = str(stored["embedder"])
        return cls._checked(ids, matrix, embedder_name)

    @classmethod
    def _checked(cls, ids: tuple[str, ...], vectors: Vectors, embedder_name: str) -> DenseIndex:
        if len(vectors) != len(ids):
            raise ValueError(f"{len(vectors)} vectors for {len(ids)} chunks; rows would misalign")
        if len(vectors) and not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
            raise ValueError("un-normalised vectors; cosine would be wrong")
        return cls(
            ids=ids,
            matrix=np.ascontiguousarray(vectors, dtype=np.float32),
            embedder_name=embedder_name,
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


def cache_file(inventory: Inventory, embedder: Embedder, cache_dir: Path = CACHE_DIR) -> Path:
    """Where this inventory's vectors are kept, keyed by the embedder and every chunk's id and text.

    The embedder's key includes the code that embeds a passage, so another model, a re-chunk, a
    re-ingest or an edit to `embed_documents` each name a new file and stale vectors are never read.
    """
    key = hashlib.sha256()
    key.update(inspect.getsource(type(embedder).embed_documents).encode())
    key.update(f"{embedder.name}\0{embedder.dimensions}".encode())
    for chunk in inventory.chunks:
        key.update(f"\0{chunk.id}\0{chunk.text}".encode())
    return cache_dir / f"index-{inventory.name}-{key.hexdigest()[:16]}.npz"
