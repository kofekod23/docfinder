# tests/v2/test_rerank_client.py
import httpx
import pytest

from server.rerank_client import RemoteReranker, RemoteRerankerError


def test_init_requires_url_and_token(monkeypatch):
    monkeypatch.delenv("COLAB_RERANK_URL", raising=False)
    monkeypatch.delenv("COLAB_QUERY_TOKEN", raising=False)
    with pytest.raises(RemoteRerankerError):
        RemoteReranker()


def test_rerank_returns_scores(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Auth-Token"] == "tok"
        body = request.content.decode()
        assert '"query":"q"' in body
        return httpx.Response(200, json={"scores": [0.2, 0.9]})

    transport = httpx.MockTransport(handler)
    r = RemoteReranker(base_url="https://x", token="tok")
    r._client = httpx.Client(transport=transport, timeout=5.0)
    assert r.rerank("q", ["a", "b"]) == [0.2, 0.9]


def test_rerank_empty_shortcircuits():
    r = RemoteReranker(base_url="https://x", token="tok")
    assert r.rerank("q", []) == []


def test_rerank_http_error_raises(monkeypatch):
    transport = httpx.MockTransport(
        lambda req: httpx.Response(500, text="boom"),
    )
    r = RemoteReranker(base_url="https://x", token="tok")
    r._client = httpx.Client(transport=transport, timeout=5.0)
    with pytest.raises(RemoteRerankerError):
        r.rerank("q", ["a"])
