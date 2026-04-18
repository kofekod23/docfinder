# tests/v2/test_query_server_rerank.py
from fastapi.testclient import TestClient

import colab.query_server as qs


class _StubRe:
    def rerank(self, pairs):
        return [float(i) for i in range(len(pairs))]


class _StubWrapper:
    _model = object()


def _prep_env(monkeypatch):
    monkeypatch.setenv("COLAB_QUERY_TOKEN", "secret")
    monkeypatch.setenv("LOAD_RERANKER", "0")
    monkeypatch.setattr(qs, "_wrapper", _StubWrapper(), raising=False)


def _client(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setattr(qs, "_reranker", _StubRe(), raising=False)
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
    _prep_env(monkeypatch)
    monkeypatch.setattr(qs, "_reranker", None, raising=False)
    c = TestClient(qs.app)
    resp = c.post(
        "/rerank",
        headers={"X-Auth-Token": "secret"},
        json={"query": "q", "documents": ["a"]},
    )
    assert resp.status_code == 503
