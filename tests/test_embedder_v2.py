from unittest.mock import MagicMock, patch
import numpy as np
import pytest


@pytest.fixture
def fake_model_out():
    return {
        "dense_vecs": np.zeros((2, 1024), dtype=np.float32),
        "lexical_weights": [
            {10: 0.9, 20: 0.8, 30: 0.1},
            {10: 0.5, 40: 0.7},
        ],
        "colbert_vecs": [
            np.zeros((3, 1024), dtype=np.float32),
            np.zeros((2, 1024), dtype=np.float32),
        ],
    }


def test_encode_chunks_returns_parallel_lists(fake_model_out):
    with patch("colab.embedder_v2._build_model") as build:
        m = MagicMock()
        m.encode.return_value = fake_model_out
        m.tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"
        build.return_value = m
        from colab.embedder_v2 import BGEM3Wrapper
        w = BGEM3Wrapper()
        result = w.encode(["aa", "bb"])
        assert len(result.dense) == 2
        assert len(result.sparse) == 2
        assert len(result.colbert) == 2
        assert result.sparse[0] == ([10, 20, 30], [0.9, 0.8, 0.1])


def test_keywords_from_chunk_filters_and_topk(fake_model_out):
    from colab.embedder_v2 import keywords_from_weights
    weights = {10: 0.9, 20: 0.8, 30: 0.1, 40: 0.05}
    # tokenizer returns "tok{id}" except id=30 returns "1" (numeric)
    def decode(ids):
        tid = ids[0]
        return "1" if tid == 30 else f"tok{tid}"
    kws = keywords_from_weights(weights, decode=decode, top_k=10)
    assert "1" not in kws  # numeric-only filtered
    assert kws[:2] == ["tok10", "tok20"]


def test_aggregate_doc_keywords_weighted_by_chunk_length():
    from colab.embedder_v2 import aggregate_doc_keywords
    chunks = [
        ({10: 0.5, 20: 0.5}, 10),   # weights, chunk_token_count
        ({10: 0.1, 30: 0.9}, 100),
    ]
    def decode(ids):
        return f"tok{ids[0]}"
    out = aggregate_doc_keywords(chunks, decode=decode, top_k=3)
    # tok30 should dominate because of the 100-token chunk
    assert out[0] == "tok30"
