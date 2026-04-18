"""Tests unitaires du wrapper Qwen3-Embedding (modèle factice)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from colab.qwen_embedder import QwenEmbedderWrapper


def _fake_model(dim: int = 8) -> MagicMock:
    model = MagicMock()

    def encode(texts, batch_size=32, max_length=512, **_):
        return np.array([[float(i) / dim] * dim for i in range(len(texts))])

    model.encode.side_effect = encode
    return model


def test_encode_returns_dense_only() -> None:
    wrapper = QwenEmbedderWrapper(model=_fake_model(dim=4))
    result = wrapper.encode(["bonjour", "monde"])
    assert len(result.dense) == 2
    assert len(result.dense[0]) == 4
    assert result.sparse == []
    assert result.colbert == []
    assert result.lexical_weights == []


def test_encode_empty_input() -> None:
    wrapper = QwenEmbedderWrapper(model=_fake_model(dim=4))
    result = wrapper.encode([])
    assert result.dense == []
    assert result.sparse == []
    assert result.colbert == []


def test_encode_passes_batch_size() -> None:
    model = _fake_model(dim=4)
    wrapper = QwenEmbedderWrapper(model=model)
    wrapper.encode(["a", "b", "c"], batch_size=16, max_length=256)
    call_kwargs = model.encode.call_args.kwargs
    assert call_kwargs["batch_size"] == 16
    assert call_kwargs["max_length"] == 256
