from pathlib import Path
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"x" * 100)
    (root / "big.pdf").write_bytes(b"x" * (60 * 1024 * 1024))  # 60 MB
    (root / "note.md").write_bytes(b"# hello\n")
    monkeypatch.setenv("DOCFINDER_ROOTS", str(root))

    from server.files_api import router as files_router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(files_router)
    return TestClient(app), root


def test_files_list_returns_metadata(client):
    c, root = client
    r = c.get("/files/list", params={"root": str(root)})
    assert r.status_code == 200
    items = r.json()
    by_path = {item["path"]: item for item in items}
    assert "a.pdf" in by_path
    assert by_path["a.pdf"]["size"] == 100
    assert "doc_id" in by_path["a.pdf"]
    assert isinstance(by_path["a.pdf"]["mtime"], int)


def test_files_list_filters_files_over_50mb(client):
    c, root = client
    r = c.get("/files/list", params={"root": str(root)})
    items = r.json()
    paths = {item["path"] for item in items}
    # 60 MB file > 50 MB threshold → flagged filename_only, but still listed
    assert "big.pdf" in paths
    big = next(i for i in items if i["path"] == "big.pdf")
    assert big["mode"] == "filename_only"
    small = next(i for i in items if i["path"] == "a.pdf")
    assert small["mode"] == "full_or_head"


def test_files_list_rejects_root_outside_allowed(client, tmp_path: Path):
    c, _ = client
    r = c.get("/files/list", params={"root": str(tmp_path / "other")})
    assert r.status_code == 400
