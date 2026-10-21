"""The local models' setup, against fake sentence-transformers classes: no weights are loaded."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sentence_transformers
import torch

from aigent.config import EMBED_MODEL, EMBED_QUERY_PREFIXES, LOCAL_MODEL_DTYPE
from aigent.retrieval.embed import LocalEmbedder, query_prefix_of
from aigent.retrieval.rerank import LocalReranker

BGE = EMBED_QUERY_PREFIXES["BAAI/bge-large-en-v1.5"]


@pytest.mark.parametrize(
    ("model", "declared", "prefix"),
    [
        pytest.param(
            "Qwen/Qwen3-Embedding-0.6B",
            "Instruct: q\nQuery:",
            "Instruct: q\nQuery:",
            id="a declared prefix wins",
        ),
        pytest.param("BAAI/bge-large-en-v1.5", None, BGE, id="an undeclared one comes from config"),
        pytest.param("BAAI/bge-large-en-v1.5", "", BGE, id="an empty declaration is none"),
        pytest.param("Alibaba-NLP/gte-modernbert-base", None, "", id="a model that wants none"),
    ],
)
def test_a_models_query_prefix_is_its_own(model: str, declared: str | None, prefix: str) -> None:
    assert query_prefix_of(model, declared) == prefix


def test_a_model_with_no_known_prefix_is_refused_not_given_another_models() -> None:
    with pytest.raises(ValueError, match="EMBED_QUERY_PREFIXES"):
        query_prefix_of("someone/unknown-embedder", None)


def test_the_configured_embedder_has_a_known_prefix() -> None:
    assert isinstance(query_prefix_of(EMBED_MODEL, None), str), "no ValueError: it is listed"


class _Loaded:
    """Records how a model was loaded and stands in for it."""

    calls: list[tuple[str, dict[str, object]]] = []
    prompts: dict[str, str] = {}
    max_seq_length = 512

    def __init__(self, name: str, **kwargs: object) -> None:
        _Loaded.calls.append((name, kwargs))

    def get_embedding_dimension(self) -> int:
        return 4


@pytest.mark.parametrize(
    ("attribute", "build"),
    [
        pytest.param(
            "SentenceTransformer",
            lambda: LocalEmbedder("BAAI/bge-large-en-v1.5", device="cpu"),
            id="the embedder",
        ),
        pytest.param(
            "CrossEncoder",
            lambda: LocalReranker("BAAI/bge-reranker-v2-m3", device="cpu"),
            id="the reranker",
        ),
    ],
)
def test_a_local_model_loads_at_full_precision_whatever_it_declares(
    monkeypatch: pytest.MonkeyPatch, attribute: str, build: Callable[[], object]
) -> None:
    monkeypatch.setattr(sentence_transformers, attribute, _Loaded)
    _Loaded.calls = []

    build()

    [(_, kwargs)] = _Loaded.calls
    assert kwargs["model_kwargs"] == {"dtype": getattr(torch, LOCAL_MODEL_DTYPE)}, kwargs
    assert LOCAL_MODEL_DTYPE == "float32"
