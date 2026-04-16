from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def app_and_mock():
    from server import admin_v2
    qdrant = MagicMock()
    admin_v2.set_qdrant_client(qdrant, collection="docfinder_v2")
    app = FastAPI()
    app.include_router(admin_v2.router)
    return TestClient(app), qdrant


def test_indexed_state_returns_mapping(app_and_mock):
    client, qdrant = app_and_mock

    class P:
        def __init__(self, doc_id, mtime):
            self.payload = {"doc_id": doc_id, "mtime": mtime}

    # scroll returns (points, next_page_offset)
    qdrant.scroll.side_effect = [
        ([P("a", 1), P("a", 1), P("b", 2)], "tok"),
        ([P("c", 3)], None),
    ]
    r = client.post("/admin/indexed-state", json={})
    assert r.status_code == 200
    assert r.json() == {"a": 1, "b": 2, "c": 3}


def test_delete_doc_removes_points_by_doc_id(app_and_mock):
    client, qdrant = app_and_mock
    r = client.delete("/admin/doc/abc123")
    assert r.status_code == 200
    assert qdrant.delete.called
    called_filter = qdrant.delete.call_args.kwargs["points_selector"]
    # filter object's json form should mention abc123
    assert "abc123" in str(called_filter)


def test_progress_post_then_get(app_and_mock):
    client, _ = app_and_mock
    body = {"total": 10, "done": 3, "failed": 0, "current_doc": "x.pdf",
            "gpu_util_pct": 80, "vram_used_mb": 9000,
            "chunks_per_sec": 30.0, "eta_seconds": 120,
            "stage_counts": {"downloaded": 3}}
    r = client.post("/admin/progress", json=body)
    assert r.status_code == 200
    got = client.get("/admin/progress").json()
    assert got["done"] == 3
    assert got["current_doc"] == "x.pdf"
