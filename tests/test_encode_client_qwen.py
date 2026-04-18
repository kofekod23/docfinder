"""Tests du RemoteEncoder avec sélection d'embedder (bgem3 vs qwen)."""

from __future__ import annotations

import httpx
import pytest

from server.encode_client import RemoteEncoder


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("COLAB_ENCODE_URL", "https://colab.example")
    monkeypatch.setenv("COLAB_QUERY_TOKEN", "tok")


def test_default_embedder_hits_encode():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"dense": [[0.1]], "sparse": [], "colbert": []})

    transport = httpx.MockTransport(handler)
    enc = RemoteEncoder(embedder="bgem3")
    enc._client = httpx.Client(transport=transport)
    enc._url = "https://colab.example/encode"

    enc.encode(["hi"])
    assert captured["url"].endswith("/encode")


def test_qwen_embedder_hits_encode_qwen():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"dense": [[0.1]], "sparse": [], "colbert": []})

    transport = httpx.MockTransport(handler)
    enc = RemoteEncoder(embedder="qwen")
    enc._client = httpx.Client(transport=transport)
    enc._url = "https://colab.example/encode_qwen"

    enc.encode(["hi"])
    assert captured["url"].endswith("/encode_qwen")
