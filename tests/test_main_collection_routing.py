"""Vérifie que /search lit DOCFINDER_COLLECTION et DOCFINDER_EMBEDDER."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("USE_V2", "true")
    monkeypatch.setenv("DOCFINDER_COLLECTION", "docfinder_v2_qwen")
    monkeypatch.setenv("DOCFINDER_EMBEDDER", "qwen")
    monkeypatch.setenv("COLAB_ENCODE_URL", "https://colab.example")
    monkeypatch.setenv("COLAB_QUERY_TOKEN", "tok")


def test_resolve_collection_prefers_env():
    from server.main import _resolve_search_config
    cfg = _resolve_search_config()
    assert cfg["collection"] == "docfinder_v2_qwen"
    assert cfg["embedder"] == "qwen"


def test_resolve_collection_fallback_default(monkeypatch):
    monkeypatch.delenv("DOCFINDER_COLLECTION", raising=False)
    monkeypatch.delenv("DOCFINDER_EMBEDDER", raising=False)
    from server.main import _resolve_search_config
    cfg = _resolve_search_config()
    assert cfg["collection"] == "docfinder_v2"
    assert cfg["embedder"] == "bgem3"
