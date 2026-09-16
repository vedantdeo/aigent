"""Turning text into vectors with a local model.

Anthropic ships no embedding model, so this is the one part of Entropic running on someone else's
weights. Queries take an instruction prefix and passages do not; vectors come out L2-normalised;
`count_truncated` reports what the model's token limit actually cut.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from entropic.config import EMBED_BATCH, EMBED_MODEL, EMBED_QUERY_PREFIX

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    """What the store needs from an embedding model, and nothing more.

    A protocol so tests can substitute a fake without importing torch.
    """

    name: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors: ...


class LocalEmbedder:
    """`sentence-transformers` on whatever device this machine has, defaulting to MPS.

    `dimensions`, `max_tokens` and `query_prefix` come off the loaded model where it declares them.
    """

    def __init__(self, model_name: str = EMBED_MODEL, *, device: str | None = None) -> None:
        """Load the model. `device=None` picks MPS when available, else CPU.

        torch and SentenceTransformer are imported here so the rest of the package stays cheap to
        import. `query_prefix` falls back to `EMBED_QUERY_PREFIX` for models too old to declare one.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)
        self.dimensions = cast(int, self._model.get_embedding_dimension())
        self.max_tokens = cast(int, self._model.max_seq_length)
        self.batch_size = EMBED_BATCH
        self.query_prefix = self._model.prompts.get("query") or EMBED_QUERY_PREFIX

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """One row per text, L2-normalised, in order. No query prefix — that side is asymmetric."""
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True
        )
        return cast(Vectors, vectors)

    def embed_query(self, text: str) -> Vectors:
        """A single query vector, with the instruction prefix. One dimension, not two."""
        query_text = f"{self.query_prefix}{text}"
        vector = self._model.encode([query_text], normalize_embeddings=True)
        return cast(Vectors, vector[0])

    def count_truncated(self, texts: Sequence[str]) -> int:
        """How many of these the model will read only part of.

        Text past the token limit is not read, while the chunk goes on claiming to contain it.
        """
        if not texts:
            return 0
        tokenised = self._model.tokenizer(list(texts), add_special_tokens=True)["input_ids"]
        return sum(len(ids) > self.max_tokens for ids in tokenised)
