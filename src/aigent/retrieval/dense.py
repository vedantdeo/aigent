"""The dense index: brute-force cosine search over a matrix of chunk vectors.

No approximate index and no database: a dot product over NumPy is exact and fast enough for this
corpus, so a miss belongs to the chunking or the embedding rather than to an index's own recall
knob. The index holds ids and vectors; `Inventory` holds the chunks. `build_cached` keeps every
vector on disk in one store per embedder, keyed by its text, so any index embeds only text the
store has not seen. Progress goes to this module's logger.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aigent.config import EMBED_LOG_EVERY, TOP_K
from aigent.retrieval.chunk import Inventory
from aigent.retrieval.corpus import CACHE_DIR
from aigent.retrieval.embed import Embedder, Vectors
from aigent.retrieval.hits import Hit

log = logging.getLogger(__name__)


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def embedded(texts: list[str], embedder: Embedder, name: str) -> Vectors:
    """`texts`' vectors, `EMBED_LOG_EVERY` at a time, logging each slice under `name`."""
    started = time.perf_counter()
    parts: list[Vectors] = [embedder.embed_documents([])]
    for start in range(0, len(texts), EMBED_LOG_EVERY):
        parts.append(embedder.embed_documents(texts[start : start + EMBED_LOG_EVERY]))
        done = min(start + EMBED_LOG_EVERY, len(texts))
        log.info(
            "%s: embedded %s/%s (%d%%) with %s, %s",
            name,
            f"{done:,}",
            f"{len(texts):,}",
            100 * done // len(texts),
            embedder.name,
            _clock(time.perf_counter() - started),
        )
    return np.concatenate(parts).astype(np.float32, copy=False)


def text_key(text: str) -> str:
    """A text's key in the vector store: its SHA-256, which no two texts here share."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def store_file(embedder: Embedder, cache_dir: Path = CACHE_DIR) -> Path:
    """Where `embedder`'s vectors are kept: one file per model and per version of the code that
    embeds a passage, so another model or an edit to `embed_documents` never reads stale vectors."""
    key = hashlib.sha256()
    key.update(inspect.getsource(type(embedder).embed_documents).encode())
    key.update(f"{embedder.name}\0{embedder.dimensions}".encode())
    return cache_dir / f"vectors-{_safe(embedder.name)}-{key.hexdigest()[:16]}.npz"


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def read_store(path: Path) -> tuple[list[str], Vectors]:
    """The keys and vectors `write_store` wrote, or none when the file is missing or damaged."""
    try:
        with np.load(path, allow_pickle=False) as stored:
            keys = [str(key) for key in stored["keys"]]
            matrix = np.asarray(stored["matrix"], dtype=np.float32)
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile):
        return [], np.empty((0, 0), dtype=np.float32)
    if len(keys) != len(matrix) or len(set(keys)) != len(keys):
        return [], np.empty((0, 0), dtype=np.float32)
    return keys, matrix


def write_store(path: Path, keys: list[str], matrix: Vectors) -> None:
    """Keys and vectors to `path`, whole or not at all, replacing an older version's store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with partial.open("wb") as handle:  # a file, since `np.savez` adds `.npz` to a bare path
        np.savez(handle, keys=np.array(keys, dtype=np.str_), matrix=matrix)
    partial.replace(path)
    prefix = path.name.rsplit("-", 1)[0]
    for stale in path.parent.glob(f"{prefix}-*.npz"):
        if stale != path:
            stale.unlink(missing_ok=True)


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
        """Embed every chunk, ids and rows aligned, `EMBED_LOG_EVERY` at a time, logging each.

        Refuses a vector count that does not match the inventory, and un-normalised rows — both
        would leave every score plausible and every citation wrong.
        """
        vectors = embedded(inventory.texts(), embedder, inventory.name)
        return cls._checked(tuple(inventory.ids()), vectors, embedder.name)

    @classmethod
    def build_cached(
        cls, inventory: Inventory, embedder: Embedder, cache_dir: Path = CACHE_DIR
    ) -> DenseIndex:
        """`build`, embedding only the texts `embedder`'s store has not seen, then adding them.
        A missing or damaged store is started afresh, never raised on."""
        path = store_file(embedder, cache_dir)
        keys, matrix = read_store(path)
        row = {key: n for n, key in enumerate(keys)}
        wanted = [text_key(text) for text in inventory.texts()]
        new = {key: text for key, text in zip(wanted, inventory.texts(), strict=True)}
        new = {key: text for key, text in new.items() if key not in row}
        log.info(
            "%s: %s of %s texts already embedded with %s; embedding %s",
            inventory.name,
            f"{len(wanted) - sum(key in new for key in wanted):,}",
            f"{len(wanted):,}",
            embedder.name,
            f"{len(new):,}",
        )
        if new:
            fresh = embedded(list(new.values()), embedder, inventory.name)
            matrix = fresh if not keys else np.concatenate([matrix, fresh])
            for key in new:
                row[key] = len(keys)
                keys.append(key)
            write_store(path, keys, matrix)
        rows = [row[key] for key in wanted]
        return cls._checked(tuple(inventory.ids()), matrix[rows], embedder.name)

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
