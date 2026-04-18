"""Tests d'intégration FastAPI pour l'endpoint /encode_qwen."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from colab.embedder_v2 import EncodeResult
from colab import query_server


def test_encode_qwen_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_QUERY_TOKEN", "test-token")
    wrapper = MagicMock()
    monkeypatch.setattr(query_server, "_qwen_wrapper", wrapper, raising=False)
    client = TestClient(query_server.app)
    resp = client.post("/encode_qwen", json={"queries": ["hi"]})
    assert resp.status_code == 401


def test_encode_qwen_returns_dense_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_QUERY_TOKEN", "test-token")
    wrapper = MagicMock()
    wrapper.encode.return_value = EncodeResult(
        dense=[[0.1, 0.2, 0.3]],
        sparse=[],
        colbert=[],
        lexical_weights=[],
    )
    monkeypatch.setattr(query_server, "_qwen_wrapper", wrapper, raising=False)
    client = TestClient(query_server.app)
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


def test_encode_qwen_503_when_wrapper_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_QUERY_TOKEN", "test-token")
    monkeypatch.setattr(query_server, "_qwen_wrapper", None, raising=False)
    client = TestClient(query_server.app)
    resp = client.post(
        "/encode_qwen",
        json={"queries": ["hi"]},
        headers={"X-Auth-Token": "test-token"},
    )
    assert resp.status_code == 503
