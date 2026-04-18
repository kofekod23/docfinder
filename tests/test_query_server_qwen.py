"""Tests d'intégration FastAPI pour l'endpoint /encode_qwen."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from colab.embedder_v2 import EncodeResult
from colab import query_server


def _make_client(wrapper: MagicMock) -> TestClient:
    os.environ["COLAB_QUERY_TOKEN"] = "test-token"
    query_server.set_qwen_wrapper(wrapper)
    return TestClient(query_server.app)


def test_encode_qwen_requires_token() -> None:
    wrapper = MagicMock()
    client = _make_client(wrapper)
    resp = client.post("/encode_qwen", json={"queries": ["hi"]})
    assert resp.status_code == 401


def test_encode_qwen_returns_dense_only() -> None:
    wrapper = MagicMock()
    wrapper.encode.return_value = EncodeResult(
        dense=[[0.1, 0.2, 0.3]],
        sparse=[],
        colbert=[],
        lexical_weights=[],
    )
    client = _make_client(wrapper)
    resp = client.post(
        "/encode_qwen",
        json={"queries": ["bonjour"]},
        headers={"X-Auth-Token": "test-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dense"] == [[0.1, 0.2, 0.3]]
    assert body["sparse"] == []
    assert body["colbert"] == []


def test_encode_qwen_503_when_wrapper_missing() -> None:
    os.environ["COLAB_QUERY_TOKEN"] = "test-token"
    query_server._qwen_wrapper = None
    client = TestClient(query_server.app)
    resp = client.post(
        "/encode_qwen",
        json={"queries": ["hi"]},
        headers={"X-Auth-Token": "test-token"},
    )
    assert resp.status_code == 503
