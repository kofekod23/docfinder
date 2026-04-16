from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys


def test_search_endpoint_routes_to_v2_when_flag_on(monkeypatch):
    monkeypatch.setenv("USE_V2", "true")
    # Clear module cache to ensure fresh import with env var
    if "server.main" in sys.modules:
        del sys.modules["server.main"]

    with patch("server.search.search_v2") as mocked_v2:
        mocked_v2.return_value = []
        from server.main import app
        with TestClient(app) as client:
            r = client.post("/search", json={"query": "hello", "limit": 5})
            assert r.status_code == 200
            assert mocked_v2.called


def test_search_endpoint_routes_to_v1_when_flag_off(monkeypatch):
    monkeypatch.setenv("USE_V2", "false")
    # Clear module cache to ensure fresh import with env var
    if "server.main" in sys.modules:
        del sys.modules["server.main"]

    from server.main import app
    import server.main

    with patch.object(server.main, "_engine_search", return_value=[]) as mocked_v1:
        with TestClient(app) as client:
            r = client.post("/search", json={"query": "hello", "limit": 5})
            assert r.status_code == 200
            assert mocked_v1.called
