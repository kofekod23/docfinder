# tests/test_colab_client.py
import httpx
import pytest

from colab.client import MacClient


def test_list_files():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/files/list"
        return httpx.Response(200, json=[{"path": "a.pdf", "size": 10,
                                           "mtime": 1, "doc_id": "d",
                                           "mode": "full_or_head",
                                           "abs_path": "/x/a.pdf"}])
    t = httpx.MockTransport(handler)
    c = MacClient("https://mac.example", transport=t)
    items = c.list_files(root="/x")
    assert items[0]["doc_id"] == "d"


def test_download_bytes():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/files/raw"
        return httpx.Response(200, content=b"payload")
    t = httpx.MockTransport(handler)
    c = MacClient("https://mac.example", transport=t)
    assert c.download("/x/a.pdf") == b"payload"


def test_upsert_chunks_posts_json():
    seen = {}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True})
    t = httpx.MockTransport(handler)
    c = MacClient("https://mac.example", transport=t)
    r = c.upsert_chunks_v2([{"id": "x"}])
    assert r["ok"] is True
    assert seen["path"] == "/admin/upsert-v2"


def test_push_progress():
    seen = {}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        return httpx.Response(200, json={"ok": True})
    t = httpx.MockTransport(handler)
    c = MacClient("https://mac.example", transport=t)
    c.push_progress({"total": 1, "done": 0, "failed": 0})
    assert seen["path"] == "/admin/progress"
