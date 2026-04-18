# tests/v2/test_query_server_rerank.py
import os
from fastapi.testclient import TestClient

import colab.query_server as qs


def _client(monkeypatch):
    monkeypatch.setenv("COLAB_QUERY_TOKEN", "secret")
    class StubRe:
        def rerank(self, pairs):
            return [float(i) for i in range(len(pairs))]
    qs._reranker = StubRe()
    return TestClient(qs.app)


def test_rerank_endpoint_ok(monkeypatch):
    c = _client(monkeypatch)
    resp = c.post(
        "/rerank",
        headers={"X-Auth-Token": "secret"},
        json={"query": "q", "documents": ["a", "b", "c"]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"scores": [0.0, 1.0, 2.0]}


def test_rerank_requires_token(monkeypatch):
    c = _client(monkeypatch)
    resp = c.post("/rerank", json={"query": "q", "documents": ["a"]})
    assert resp.status_code == 401


def test_rerank_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setenv("COLAB_QUERY_TOKEN", "secret")
    qs._reranker = None
    c = TestClient(qs.app)
    resp = c.post(
        "/rerank",
        headers={"X-Auth-Token": "secret"},
        json={"query": "q", "documents": ["a"]},
    )
    assert resp.status_code == 503
