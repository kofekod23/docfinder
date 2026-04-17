# Colab Pipeline v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild DocFinder's indexation pipeline so all heavy compute runs on Colab T4 while the Mac becomes a stateless file server — upgrade to BGE-M3 (dense + sparse + ColBERT), paragraph-based chunking, GPU OCR, asyncio parallelism, RRF + ColBERT rerank.

**Architecture:** Mac (FastAPI + Qdrant) serves file bytes + indexed state. Colab downloads → extracts → OCRs → chunks → embeds with BGE-M3 → upserts into a **new** Qdrant collection `docfinder_v2` (the old `docfinder` stays untouched behind feature flag `USE_V2`). Search endpoint embeds the query once, runs dense + sparse in parallel, fuses with RRF, then reranks top-50 with ColBERT MaxSim.

**Tech Stack:** Python 3.10+, FastAPI, Qdrant 1.10+ (multi-vec + sparse + prefetch), FlagEmbedding (BGE-M3), easyocr, PyMuPDF, python-docx, httpx, asyncio, pytest, Cloudflare Tunnel.

**Spec:** `docs/superpowers/specs/2026-04-16-colab-pipeline-v2-design.md`

---

## File Structure

New files (target layout):

```
docfinder/
├── shared/
│   ├── schema.py              # (modify) add DocumentChunkV2, ProgressReport, IndexedState
│   ├── hashing.py             # (new)    doc_id + file_hash helpers
│   └── chunking.py            # (new)    paragraph chunker (pure, unit tested)
├── server/
│   ├── main.py                # (modify) register v2 routers, keep old endpoints behind flag
│   ├── search.py              # (modify) USE_V2 branch → BGE-M3 + RRF + ColBERT rerank
│   ├── files_api.py           # (new)    /files/list, /files/raw (stateless file server)
│   ├── admin_v2.py            # (new)    /admin/indexed-state, /admin/doc/{id}, /admin/progress
│   └── templates/
│       └── admin.html         # (modify) v2 progress widgets (GPU bargraph, ETA, current doc)
├── colab/                     # (new)    Colab-side Python (extracted from notebook for testing)
│   ├── __init__.py
│   ├── extractor.py           # (new)    size-aware extraction (PDF/docx/md/txt) + OCR trigger
│   ├── ocr.py                 # (new)    easyocr wrapper (lazy-loaded reader)
│   ├── embedder_v2.py         # (new)    BGE-M3 wrapper (dense + sparse + colbert + keywords)
│   ├── pipeline.py            # (new)    asyncio 3-stage pipeline + flush + checkpoint
│   └── client.py              # (new)    Mac HTTP client (list, raw, upsert, progress)
├── scripts/
│   └── setup_qdrant_v2.py     # (new)    creates docfinder_v2 collection + payload indexes
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # (new)    fixtures (tmp qdrant, sample docs)
│   ├── test_hashing.py        # (new)    hashing helpers
│   ├── test_chunking.py       # (new)    paragraph chunker
│   ├── test_files_api.py      # (new)    /files/list size filter, /files/raw streaming
│   ├── test_admin_v2.py       # (new)    /admin/indexed-state, DELETE, /admin/progress
│   ├── test_extractor.py      # (new)    size-aware extraction decision
│   ├── test_embedder_v2.py    # (new)    mocked BGE-M3 keywords extraction
│   ├── test_search_v2.py      # (new)    RRF fusion + ColBERT rerank query shape
│   └── fixtures/              # (new)    sample pdf/docx/scanned pdf
└── colab_indexer.ipynb        # (modify) thin wrapper around colab/pipeline.py
```

Old files kept untouched (rollback path): `server/chunks.py`, `server/indexer.py`, `shared/embedder.py` (e5-large). Deleted only in **Task 15 (cleanup)** once v2 is validated.

---

## Task 1: Hashing helpers (doc_id + file_hash)

**Files:**
- Create: `shared/hashing.py`
- Test:   `tests/test_hashing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hashing.py
from pathlib import Path
import hashlib
import pytest

from shared.hashing import doc_id_for, file_hash_for


def test_doc_id_is_sha1_of_abs_path():
    abs_path = "/Users/julien/Documents/a.pdf"
    expected = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()
    assert doc_id_for(abs_path) == expected


def test_doc_id_stable_for_same_path():
    assert doc_id_for("/a/b.pdf") == doc_id_for("/a/b.pdf")


def test_doc_id_differs_for_different_paths():
    assert doc_id_for("/a/b.pdf") != doc_id_for("/a/c.pdf")


def test_file_hash_reads_first_mb_plus_size(tmp_path: Path):
    p = tmp_path / "sample.bin"
    payload = b"x" * (2 * 1024 * 1024)  # 2 MB
    p.write_bytes(payload)
    expected_head = payload[: 1024 * 1024]
    expected_size = p.stat().st_size
    h = hashlib.sha1()
    h.update(expected_head)
    h.update(str(expected_size).encode("utf-8"))
    assert file_hash_for(p) == h.hexdigest()


def test_file_hash_small_file(tmp_path: Path):
    p = tmp_path / "small.txt"
    p.write_bytes(b"hello")
    h = hashlib.sha1()
    h.update(b"hello")
    h.update(b"5")
    assert file_hash_for(p) == h.hexdigest()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_hashing.py -v
```
Expected: `ModuleNotFoundError: No module named 'shared.hashing'`

- [ ] **Step 3: Implement the helpers**

```python
# shared/hashing.py
"""Hashing helpers for doc_id and file_hash (see spec §4.2)."""
from __future__ import annotations

import hashlib
from pathlib import Path

HEAD_BYTES = 1024 * 1024  # 1 MB


def doc_id_for(abs_path: str) -> str:
    """SHA1 of the absolute path. Stable as long as the file is not moved."""
    return hashlib.sha1(abs_path.encode("utf-8")).hexdigest()


def file_hash_for(path: Path) -> str:
    """SHA1 of (first 1 MB of bytes || str(size)). Used to detect duplicates."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(HEAD_BYTES)
    h = hashlib.sha1()
    h.update(head)
    h.update(str(size).encode("utf-8"))
    return h.hexdigest()
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_hashing.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add shared/hashing.py tests/test_hashing.py
git commit -m "feat(hashing): add doc_id + file_hash helpers"
```

---

## Task 2: Pydantic v2 schemas

**Files:**
- Modify: `shared/schema.py` (append, do not touch existing)
- Test:   `tests/test_schema_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema_v2.py
import pytest
from pydantic import ValidationError

from shared.schema import DocumentChunkV2, ProgressReport, IndexedStateRequest


def test_document_chunk_v2_minimal():
    c = DocumentChunkV2(
        doc_id="abc", path="a.pdf", abs_path="/a/a.pdf", doc_type="pdf",
        title="a", mtime=1_700_000_000, file_hash="h",
        content="hello world", keywords_chunk=["hello"], keywords_doc=["hello"],
        page_range=[1, 2], chunk_idx=0, chunk_total=1,
        dense=[0.0] * 1024,
        sparse_indices=[10, 20], sparse_values=[0.5, 0.3],
        colbert_vecs=[[0.0] * 1024, [0.0] * 1024],
    )
    assert c.doc_id == "abc"
    assert len(c.dense) == 1024
    assert len(c.sparse_indices) == len(c.sparse_values)


def test_document_chunk_v2_sparse_length_mismatch_rejected():
    with pytest.raises(ValidationError):
        DocumentChunkV2(
            doc_id="abc", path="a.pdf", abs_path="/a/a.pdf", doc_type="pdf",
            title="a", mtime=1, file_hash="h", content="x",
            keywords_chunk=[], keywords_doc=[],
            page_range=None, chunk_idx=0, chunk_total=1,
            dense=[0.0] * 1024,
            sparse_indices=[1, 2, 3], sparse_values=[0.5],
            colbert_vecs=[[0.0] * 1024],
        )


def test_progress_report_roundtrip():
    p = ProgressReport(
        total=10, done=3, failed=0, current_doc="x.pdf",
        gpu_util_pct=80, vram_used_mb=9000, chunks_per_sec=30.0,
        eta_seconds=120,
        stage_counts={"downloaded": 3, "extracted": 3, "embedded": 3},
    )
    assert p.model_dump()["stage_counts"]["downloaded"] == 3


def test_indexed_state_request_accepts_empty_filter():
    r = IndexedStateRequest(doc_ids=None)
    assert r.doc_ids is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_schema_v2.py -v
```
Expected: `ImportError: cannot import name 'DocumentChunkV2'`.

- [ ] **Step 3: Add v2 schemas (append to existing file)**

```python
# shared/schema.py  (append at bottom)

from typing import Optional, List, Dict
from pydantic import model_validator


class DocumentChunkV2(BaseModel):
    """Chunk BGE-M3 prêt à indexer dans docfinder_v2."""
    doc_id: str
    path: str
    abs_path: str
    doc_type: str
    title: str
    mtime: int
    file_hash: str
    content: str
    keywords_chunk: List[str]
    keywords_doc: List[str]
    page_range: Optional[List[int]] = None
    chunk_idx: int
    chunk_total: int
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    colbert_vecs: List[List[float]]

    @model_validator(mode="after")
    def _check_sparse(self) -> "DocumentChunkV2":
        if len(self.sparse_indices) != len(self.sparse_values):
            raise ValueError("sparse_indices and sparse_values must have same length")
        return self


class ProgressReport(BaseModel):
    """Push périodique Colab → /admin/progress (spec §13)."""
    total: int
    done: int
    failed: int
    current_doc: str = ""
    gpu_util_pct: int = 0
    vram_used_mb: int = 0
    chunks_per_sec: float = 0.0
    eta_seconds: int = 0
    stage_counts: Dict[str, int] = {}


class IndexedStateRequest(BaseModel):
    """Body optionnel pour POST /admin/indexed-state."""
    doc_ids: Optional[List[str]] = None
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_schema_v2.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add shared/schema.py tests/test_schema_v2.py
git commit -m "feat(schema): add DocumentChunkV2 + ProgressReport + IndexedStateRequest"
```

---

## Task 3: Paragraph chunker (pure function)

**Files:**
- Create: `shared/chunking.py`
- Test:   `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
import pytest

from shared.chunking import split_paragraphs, is_heading, chunk_paragraphs, Chunk


def test_split_paragraphs_collapses_blank_runs():
    text = "para one\n\n\npara two\n\npara three"
    assert split_paragraphs(text) == ["para one", "para two", "para three"]


def test_split_paragraphs_strips_and_drops_empty():
    assert split_paragraphs("  \n\n  hello  \n\n\n") == ["hello"]


@pytest.mark.parametrize("line,expected", [
    ("# Title", True),
    ("### Sub", True),
    ("####### too many", False),
    ("INTRODUCTION", True),
    ("1. Section one", True),
    ("12.  Something", True),
    ("normal paragraph", False),
    ("Not a heading.", False),
])
def test_is_heading(line, expected):
    assert is_heading(line) is expected


def _tok(text: str) -> int:
    """Fake whitespace tokenizer used in tests."""
    return len(text.split())


def test_chunk_paragraphs_min_and_max_bounds():
    paras = ["x" * 5] * 50  # each paragraph = 1 token (whitespace-split)
    chunks = chunk_paragraphs(paras, tokenize_len=_tok,
                              target_min=4, target_max=6, hard_max=8, hard_min=2)
    assert all(c.token_count >= 2 for c in chunks)
    assert all(c.token_count <= 8 for c in chunks)


def test_chunk_paragraphs_forced_break_on_heading():
    paras = ["body one", "# Heading", "body two body two body two"]
    chunks = chunk_paragraphs(paras, tokenize_len=_tok,
                              target_min=1, target_max=4, hard_max=10, hard_min=1)
    assert any(c.text.startswith("# Heading") for c in chunks)
    texts_before_heading = [c.text for c in chunks if "body one" in c.text]
    assert "# Heading" not in texts_before_heading[0]


def test_chunk_paragraphs_overlap_one_paragraph():
    paras = [f"p{i}" for i in range(6)]
    chunks = chunk_paragraphs(paras, tokenize_len=_tok,
                              target_min=2, target_max=2, hard_max=3, hard_min=1)
    # consecutive chunks should share the last paragraph of the previous one
    for prev, cur in zip(chunks, chunks[1:]):
        if prev.text.split("\n\n")[-1] == "# Heading":
            continue
        assert prev.text.split("\n\n")[-1] == cur.text.split("\n\n")[0]
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_chunking.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the chunker**

```python
# shared/chunking.py
"""Paragraph-based chunker (spec §6).

Pure function, tokenizer injected so tests can use a trivial counter and
production code can inject BGE-M3's XLM-R tokenizer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_HEADING_MD_RE = re.compile(r"^#{1,6}\s+\S")
_HEADING_NUM_RE = re.compile(r"^\d+\.\s+\S")


@dataclass(frozen=True)
class Chunk:
    text: str
    token_count: int
    para_indices: tuple[int, ...]


def split_paragraphs(text: str) -> List[str]:
    """Split on runs of blank lines, strip, drop empties."""
    return [p.strip() for p in _BLANK_LINE_RE.split(text) if p.strip()]


def is_heading(line: str) -> bool:
    """Detect markdown heading, numeric heading, or all-caps line."""
    first = line.strip().splitlines()[0] if line.strip() else ""
    if not first:
        return False
    if _HEADING_MD_RE.match(first):
        return True
    if _HEADING_NUM_RE.match(first):
        return True
    letters = [c for c in first if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters):
        return True
    return False


def chunk_paragraphs(
    paragraphs: List[str],
    tokenize_len: Callable[[str], int],
    target_min: int = 400,
    target_max: int = 600,
    hard_max: int = 800,
    hard_min: int = 100,
) -> List[Chunk]:
    """Greedy accumulator with forced break on headings and 1-para overlap."""
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_idx: List[int] = []
    buf_tokens = 0
    last_was_heading = False

    def flush() -> None:
        nonlocal buf, buf_idx, buf_tokens, last_was_heading
        if not buf:
            return
        text = "\n\n".join(buf)
        chunks.append(Chunk(text=text, token_count=buf_tokens,
                            para_indices=tuple(buf_idx)))
        # overlap: keep last paragraph unless it was a heading
        if not last_was_heading and len(buf) > 1:
            buf = [buf[-1]]
            buf_idx = [buf_idx[-1]]
            buf_tokens = tokenize_len(buf[0])
        else:
            buf, buf_idx, buf_tokens = [], [], 0
        last_was_heading = False

    for i, para in enumerate(paragraphs):
        tk = tokenize_len(para)
        heading = is_heading(para)

        if heading and buf:
            flush()

        if buf_tokens + tk > hard_max and buf:
            flush()

        buf.append(para)
        buf_idx.append(i)
        buf_tokens += tk
        last_was_heading = heading

        if buf_tokens >= target_max:
            flush()
        elif buf_tokens >= target_min:
            # look-ahead: break if next para would exceed target_max
            if i + 1 < len(paragraphs):
                nxt = tokenize_len(paragraphs[i + 1])
                if buf_tokens + nxt > target_max:
                    flush()

    if buf and buf_tokens >= hard_min:
        flush()
    elif buf and chunks:
        # append to previous chunk rather than emit a tiny one
        prev = chunks.pop()
        merged = prev.text + "\n\n" + "\n\n".join(buf)
        chunks.append(Chunk(text=merged,
                            token_count=prev.token_count + buf_tokens,
                            para_indices=prev.para_indices + tuple(buf_idx)))
    elif buf:
        # single short paragraph: emit anyway (better than losing content)
        chunks.append(Chunk(text="\n\n".join(buf),
                            token_count=buf_tokens,
                            para_indices=tuple(buf_idx)))

    return chunks
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_chunking.py -v
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add shared/chunking.py tests/test_chunking.py
git commit -m "feat(chunking): paragraph-based chunker with heading breaks + 1-para overlap"
```

---

## Task 4: Mac `/files/list` endpoint (stateless file server)

**Files:**
- Create: `server/files_api.py`
- Modify: `server/main.py` (register router)
- Test:   `tests/test_files_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_files_api.py
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_files_api.py -v
```
Expected: `ImportError` on `server.files_api`.

- [ ] **Step 3: Implement `/files/list`**

```python
# server/files_api.py
"""Stateless file server endpoints — Mac only serves bytes and metadata."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from shared.hashing import doc_id_for

router = APIRouter()

FIFTY_MB = 50 * 1024 * 1024


def _allowed_roots() -> list[Path]:
    raw = os.environ.get("DOCFINDER_ROOTS", "")
    return [Path(p).expanduser().resolve() for p in raw.split(",") if p.strip()]


def _resolve_under(root: str) -> Path:
    p = Path(root).expanduser().resolve()
    allowed = _allowed_roots()
    if not any(str(p).startswith(str(a)) for a in allowed):
        raise HTTPException(400, f"root not allowed: {p}")
    if not p.exists() or not p.is_dir():
        raise HTTPException(400, f"root not a directory: {p}")
    return p


@router.get("/files/list")
def files_list(root: str = Query(...)) -> List[dict]:
    base = _resolve_under(root)
    out: list[dict] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        size = st.st_size
        mode = "filename_only" if size > FIFTY_MB else "full_or_head"
        rel = str(p.relative_to(base))
        out.append({
            "path": rel,
            "abs_path": str(p),
            "size": size,
            "mtime": int(st.st_mtime),
            "doc_id": doc_id_for(str(p)),
            "mode": mode,
        })
    return out
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_files_api.py::test_files_list_returns_metadata \
       tests/test_files_api.py::test_files_list_filters_files_over_50mb \
       tests/test_files_api.py::test_files_list_rejects_root_outside_allowed -v
```
Expected: 3 passed.

- [ ] **Step 5: Register router in main.py**

```python
# server/main.py  (add near other app.include_router calls)
from server.files_api import router as files_router
app.include_router(files_router)
```

- [ ] **Step 6: Commit**

```bash
git add server/files_api.py server/main.py tests/test_files_api.py
git commit -m "feat(files): /files/list with size-aware mode flag"
```

---

## Task 5: Mac `/files/raw` chunked streaming

**Files:**
- Modify: `server/files_api.py`
- Test:   `tests/test_files_api.py` (extend)

- [ ] **Step 1: Add failing test**

```python
# tests/test_files_api.py  (append)
def test_files_raw_streams_bytes(client):
    c, root = client
    r = c.get("/files/raw", params={"path": str(root / "a.pdf")})
    assert r.status_code == 200
    assert r.content == b"x" * 100
    assert r.headers["content-type"] == "application/octet-stream"


def test_files_raw_rejects_path_outside_roots(client, tmp_path: Path):
    c, _ = client
    outside = tmp_path / "other.pdf"
    outside.write_bytes(b"y")
    r = c.get("/files/raw", params={"path": str(outside)})
    assert r.status_code == 400


def test_files_raw_404_if_missing(client, tmp_path: Path):
    c, root = client
    r = c.get("/files/raw", params={"path": str(root / "nope.pdf")})
    assert r.status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_files_api.py -v
```

- [ ] **Step 3: Implement**

```python
# server/files_api.py  (append)
@router.get("/files/raw")
def files_raw(path: str = Query(...)) -> StreamingResponse:
    p = Path(path).expanduser().resolve()
    allowed = _allowed_roots()
    if not any(str(p).startswith(str(a)) for a in allowed):
        raise HTTPException(400, f"path not under allowed roots: {p}")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"file not found: {p}")

    def stream():
        with p.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    return
                yield chunk

    return StreamingResponse(stream(), media_type="application/octet-stream")
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_files_api.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add server/files_api.py tests/test_files_api.py
git commit -m "feat(files): /files/raw chunked streaming with root allow-list"
```

---

## Task 6: `/admin/indexed-state` + DELETE + `/admin/progress`

**Files:**
- Create: `server/admin_v2.py`
- Modify: `server/main.py` (register)
- Test:   `tests/test_admin_v2.py`

- [ ] **Step 1: Add failing test**

```python
# tests/test_admin_v2.py
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
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_admin_v2.py -v
```

- [ ] **Step 3: Implement**

```python
# server/admin_v2.py
"""V2 admin endpoints (spec §11-§13)."""
from __future__ import annotations

from typing import Optional, Dict

from fastapi import APIRouter, HTTPException
from qdrant_client.http import models as qm

from shared.schema import IndexedStateRequest, ProgressReport

router = APIRouter()

_qdrant = None
_collection = "docfinder_v2"
_last_progress: ProgressReport | None = None


def set_qdrant_client(client, collection: str = "docfinder_v2") -> None:
    global _qdrant, _collection
    _qdrant = client
    _collection = collection


def _require_qdrant():
    if _qdrant is None:
        raise HTTPException(503, "qdrant client not initialized")
    return _qdrant


@router.post("/admin/indexed-state")
def indexed_state(body: IndexedStateRequest) -> Dict[str, int]:
    q = _require_qdrant()
    offset = None
    mtimes: dict[str, int] = {}
    while True:
        points, offset = q.scroll(
            collection_name=_collection,
            limit=10_000,
            with_payload=["doc_id", "mtime"],
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            pl = p.payload or {}
            doc_id = pl.get("doc_id")
            mt = pl.get("mtime")
            if doc_id and mt is not None:
                if doc_id not in mtimes or mt > mtimes[doc_id]:
                    mtimes[doc_id] = int(mt)
        if offset is None:
            break
    if body.doc_ids is not None:
        wanted = set(body.doc_ids)
        mtimes = {k: v for k, v in mtimes.items() if k in wanted}
    return mtimes


@router.delete("/admin/doc/{doc_id}")
def delete_doc(doc_id: str) -> dict:
    q = _require_qdrant()
    flt = qm.Filter(must=[qm.FieldCondition(key="doc_id",
                                            match=qm.MatchValue(value=doc_id))])
    q.delete(collection_name=_collection, points_selector=flt)
    return {"deleted": doc_id}


@router.post("/admin/progress")
def push_progress(body: ProgressReport) -> dict:
    global _last_progress
    _last_progress = body
    return {"ok": True}


@router.get("/admin/progress")
def get_progress() -> dict:
    if _last_progress is None:
        return {"total": 0, "done": 0, "failed": 0, "current_doc": "",
                "gpu_util_pct": 0, "vram_used_mb": 0, "chunks_per_sec": 0.0,
                "eta_seconds": 0, "stage_counts": {}}
    return _last_progress.model_dump()
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_admin_v2.py -v
```

- [ ] **Step 5: Wire into main.py**

```python
# server/main.py  (in lifespan where QdrantClient is created, after client init)
from server.admin_v2 import router as admin_v2_router, set_qdrant_client
set_qdrant_client(engine.qdrant, collection="docfinder_v2")
app.include_router(admin_v2_router)
```

- [ ] **Step 6: Commit**

```bash
git add server/admin_v2.py server/main.py tests/test_admin_v2.py
git commit -m "feat(admin): /admin/indexed-state + DELETE /admin/doc/{id} + /admin/progress"
```

---

## Task 7: Qdrant v2 collection bootstrap script

**Files:**
- Create: `scripts/setup_qdrant_v2.py`
- Test:   `tests/test_setup_qdrant_v2.py`

- [ ] **Step 1: Add failing test**

```python
# tests/test_setup_qdrant_v2.py
from unittest.mock import MagicMock
from scripts.setup_qdrant_v2 import ensure_collection


def test_ensure_collection_creates_when_missing():
    client = MagicMock()
    client.collection_exists.return_value = False
    ensure_collection(client, "docfinder_v2")
    assert client.create_collection.called
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "docfinder_v2"
    assert "dense" in kwargs["vectors_config"]
    assert "colbert" in kwargs["vectors_config"]
    assert "sparse" in kwargs["sparse_vectors_config"]


def test_ensure_collection_idempotent():
    client = MagicMock()
    client.collection_exists.return_value = True
    ensure_collection(client, "docfinder_v2")
    assert not client.create_collection.called


def test_ensure_payload_indexes():
    client = MagicMock()
    client.collection_exists.return_value = False
    ensure_collection(client, "docfinder_v2")
    keys = [c.kwargs["field_name"] for c in client.create_payload_index.call_args_list]
    assert {"doc_id", "doc_type", "mtime", "path"}.issubset(keys)
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_setup_qdrant_v2.py -v
```

- [ ] **Step 3: Implement**

```python
# scripts/setup_qdrant_v2.py
"""Create docfinder_v2 collection + payload indexes (spec §4)."""
from __future__ import annotations

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def ensure_collection(client: QdrantClient, name: str = "docfinder_v2") -> None:
    if client.collection_exists(collection_name=name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE),
            "colbert": qm.VectorParams(
                size=1024,
                distance=qm.Distance.COSINE,
                multivector_config=qm.MultiVectorConfig(
                    comparator=qm.MultiVectorComparator.MAX_SIM,
                ),
                on_disk=True,
            ),
        },
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
        },
    )
    for field, schema in [
        ("doc_id", qm.PayloadSchemaType.KEYWORD),
        ("doc_type", qm.PayloadSchemaType.KEYWORD),
        ("mtime", qm.PayloadSchemaType.INTEGER),
        ("path", qm.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema,
        )


if __name__ == "__main__":
    client = QdrantClient(
        host=os.environ.get("QDRANT_HOST", "127.0.0.1"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
    )
    ensure_collection(client)
    print("docfinder_v2 ready")
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_setup_qdrant_v2.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_qdrant_v2.py tests/test_setup_qdrant_v2.py
git commit -m "feat(qdrant): bootstrap script for docfinder_v2 collection + payload indexes"
```

---

## Task 8: Colab-side size-aware extractor

**Files:**
- Create: `colab/__init__.py`, `colab/extractor.py`
- Test:   `tests/test_extractor.py`, `tests/fixtures/sample.txt`

- [ ] **Step 1: Create fixtures + failing test**

```bash
mkdir -p colab tests/fixtures
touch colab/__init__.py
```

```python
# tests/test_extractor.py
from pathlib import Path
import pytest

from colab.extractor import decide_mode, extract_text, ExtractionResult


def test_decide_mode_filename_only_above_50mb():
    assert decide_mode(size=60 * 1024 * 1024, page_count=None) == "filename_only"


def test_decide_mode_head_only_for_large_file():
    assert decide_mode(size=25 * 1024 * 1024, page_count=None) == "head_only"


def test_decide_mode_head_only_for_long_pdf():
    assert decide_mode(size=5 * 1024 * 1024, page_count=40) == "head_only"


def test_decide_mode_full():
    assert decide_mode(size=1 * 1024 * 1024, page_count=5) == "full"


def test_extract_text_txt_full(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hello\n\nworld\n")
    res = extract_text(p, mode="full")
    assert "hello" in res.text and "world" in res.text
    assert res.doc_type == "txt"


def test_extract_text_filename_only(tmp_path: Path):
    p = tmp_path / "huge.pdf"
    p.write_bytes(b"%PDF-stub")
    res = extract_text(p, mode="filename_only")
    assert res.text.strip() == "huge.pdf"
    assert res.doc_type == "pdf"
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_extractor.py -v
```

- [ ] **Step 3: Implement the extractor (size-aware, stubs for PDF until Task 9)**

```python
# colab/extractor.py
"""Size-aware extractor (spec §5).

Decision matrix:
  size > 50 MB                        -> filename_only
  size > 20 MB OR (pdf AND pages>30)  -> head_only (first 5-10 pages)
  else                                -> full
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TWENTY_MB = 20 * 1024 * 1024
FIFTY_MB = 50 * 1024 * 1024
LONG_PDF_PAGES = 30
HEAD_PAGES = 8


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    doc_type: str
    page_count: Optional[int]
    ocr_pages: tuple[int, ...] = ()


def decide_mode(size: int, page_count: Optional[int]) -> str:
    if size > FIFTY_MB:
        return "filename_only"
    if size > TWENTY_MB or (page_count is not None and page_count > LONG_PDF_PAGES):
        return "head_only"
    return "full"


def _detect_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {"pdf": "pdf", "docx": "docx", "doc": "docx",
            "md": "md", "markdown": "md", "txt": "txt"}.get(ext, ext or "bin")


def extract_text(path: Path, mode: str) -> ExtractionResult:
    doc_type = _detect_type(path)
    if mode == "filename_only":
        return ExtractionResult(text=path.name, doc_type=doc_type, page_count=None)
    if doc_type == "txt" or doc_type == "md":
        return ExtractionResult(
            text=path.read_text(encoding="utf-8", errors="replace"),
            doc_type=doc_type, page_count=None,
        )
    if doc_type == "pdf":
        return _extract_pdf(path, mode)
    if doc_type == "docx":
        return _extract_docx(path)
    return ExtractionResult(text="", doc_type=doc_type, page_count=None)


def _extract_pdf(path: Path, mode: str) -> ExtractionResult:
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    total = doc.page_count
    limit = HEAD_PAGES if mode == "head_only" else total
    parts: list[str] = []
    ocr_pages: list[int] = []
    for i in range(min(limit, total)):
        page = doc.load_page(i)
        txt = page.get_text()
        if len(txt.strip()) < 10:
            ocr_pages.append(i)  # signalled; OCR done downstream in colab/ocr.py
            continue
        parts.append(txt)
    return ExtractionResult(
        text="\n\n".join(parts),
        doc_type="pdf",
        page_count=total,
        ocr_pages=tuple(ocr_pages),
    )


def _extract_docx(path: Path) -> ExtractionResult:
    import docx
    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return ExtractionResult(text="\n\n".join(parts), doc_type="docx", page_count=None)
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_extractor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add colab/__init__.py colab/extractor.py tests/test_extractor.py
git commit -m "feat(extractor): size-aware extraction for pdf/docx/md/txt + ocr page flag"
```

---

## Task 9: Colab OCR wrapper (easyocr, lazy-loaded)

**Files:**
- Create: `colab/ocr.py`
- Test:   `tests/test_ocr.py`

- [ ] **Step 1: Add failing test (mocked reader)**

```python
# tests/test_ocr.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


def test_ocr_page_joins_lines_with_newline():
    with patch("colab.ocr._build_reader") as build:
        reader = MagicMock()
        reader.readtext.return_value = ["bonjour", "le monde"]
        build.return_value = reader
        from colab.ocr import ocr_page, _reset_reader
        _reset_reader()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        text = ocr_page(img)
        assert text == "bonjour\nle monde"
        reader.readtext.assert_called_once()
        # reader is cached
        ocr_page(img)
        build.assert_called_once()


def test_ocr_page_empty_when_no_detections():
    with patch("colab.ocr._build_reader") as build:
        reader = MagicMock()
        reader.readtext.return_value = []
        build.return_value = reader
        from colab.ocr import ocr_page, _reset_reader
        _reset_reader()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert ocr_page(img) == ""
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_ocr.py -v
```

- [ ] **Step 3: Implement**

```python
# colab/ocr.py
"""easyocr wrapper, lazy-loaded (spec §7)."""
from __future__ import annotations

from typing import Optional
import numpy as np

_reader = None


def _build_reader():
    import easyocr
    return easyocr.Reader(["fr", "en"], gpu=True)


def _reset_reader() -> None:
    global _reader
    _reader = None


def ocr_page(image: np.ndarray) -> str:
    """Run OCR on a rendered PDF page (RGB numpy array). Join lines with \\n."""
    global _reader
    if _reader is None:
        _reader = _build_reader()
    lines = _reader.readtext(image, detail=0, paragraph=True)
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_ocr.py -v
```

- [ ] **Step 5: Commit**

```bash
git add colab/ocr.py tests/test_ocr.py
git commit -m "feat(ocr): easyocr wrapper with lazy reader init"
```

---

## Task 10: Colab BGE-M3 wrapper + keyword extraction

**Files:**
- Create: `colab/embedder_v2.py`
- Test:   `tests/test_embedder_v2.py`

- [ ] **Step 1: Failing test (mock model)**

```python
# tests/test_embedder_v2.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


@pytest.fixture
def fake_model_out():
    return {
        "dense_vecs": np.zeros((2, 1024), dtype=np.float32),
        "lexical_weights": [
            {10: 0.9, 20: 0.8, 30: 0.1},
            {10: 0.5, 40: 0.7},
        ],
        "colbert_vecs": [
            np.zeros((3, 1024), dtype=np.float32),
            np.zeros((2, 1024), dtype=np.float32),
        ],
    }


def test_encode_chunks_returns_parallel_lists(fake_model_out):
    with patch("colab.embedder_v2._build_model") as build:
        m = MagicMock()
        m.encode.return_value = fake_model_out
        m.tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"
        build.return_value = m
        from colab.embedder_v2 import BGEM3Wrapper
        w = BGEM3Wrapper()
        result = w.encode(["aa", "bb"])
        assert len(result.dense) == 2
        assert len(result.sparse) == 2
        assert len(result.colbert) == 2
        assert result.sparse[0] == ([10, 20, 30], [0.9, 0.8, 0.1])


def test_keywords_from_chunk_filters_and_topk(fake_model_out):
    from colab.embedder_v2 import keywords_from_weights
    weights = {10: 0.9, 20: 0.8, 30: 0.1, 40: 0.05}
    # tokenizer returns "tok{id}" except id=30 returns "1" (numeric)
    def decode(ids):
        tid = ids[0]
        return "1" if tid == 30 else f"tok{tid}"
    kws = keywords_from_weights(weights, decode=decode, top_k=10)
    assert "1" not in kws  # numeric-only filtered
    assert kws[:2] == ["tok10", "tok20"]


def test_aggregate_doc_keywords_weighted_by_chunk_length():
    from colab.embedder_v2 import aggregate_doc_keywords
    chunks = [
        ({10: 0.5, 20: 0.5}, 10),   # weights, chunk_token_count
        ({10: 0.1, 30: 0.9}, 100),
    ]
    def decode(ids):
        return f"tok{ids[0]}"
    out = aggregate_doc_keywords(chunks, decode=decode, top_k=3)
    # tok30 should dominate because of the 100-token chunk
    assert out[0] == "tok30"
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_embedder_v2.py -v
```

- [ ] **Step 3: Implement**

```python
# colab/embedder_v2.py
"""BGE-M3 wrapper: dense + sparse + ColBERT + keyword extraction (spec §3, §8)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Iterable

_STOPWORDS = {
    # minimal FR/EN stoplist, expand later
    "le", "la", "les", "un", "une", "des", "et", "ou", "de", "du", "au", "aux",
    "à", "en", "pour", "par", "sur", "dans", "avec", "sans",
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "with", "to", "at",
    "is", "are", "was", "were",
}


@dataclass(frozen=True)
class EncodeResult:
    dense: List[List[float]]
    sparse: List[Tuple[List[int], List[float]]]
    colbert: List[List[List[float]]]
    lexical_weights: List[Dict[int, float]]


class BGEM3Wrapper:
    def __init__(self, model=None):
        self._model = model

    def _model_or_build(self):
        if self._model is None:
            self._model = _build_model()
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32,
               max_length: int = 512) -> EncodeResult:
        m = self._model_or_build()
        out = m.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        dense = [v.tolist() for v in out["dense_vecs"]]
        sparse: List[Tuple[List[int], List[float]]] = []
        for w in out["lexical_weights"]:
            items = sorted(w.items(), key=lambda kv: -kv[1])
            idx = [int(k) for k, _ in items]
            val = [float(v) for _, v in items]
            sparse.append((idx, val))
        colbert = [[row.tolist() for row in mat] for mat in out["colbert_vecs"]]
        return EncodeResult(dense=dense, sparse=sparse, colbert=colbert,
                            lexical_weights=out["lexical_weights"])


def _build_model():
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")


def keywords_from_weights(
    weights: Dict[int, float],
    decode: Callable[[List[int]], str],
    top_k: int = 10,
    min_len: int = 3,
) -> List[str]:
    items = sorted(weights.items(), key=lambda kv: -kv[1])
    out: list[str] = []
    seen: set[str] = set()
    for tid, _ in items:
        tok = decode([tid]).strip()
        if not tok or tok.startswith("##") or tok.startswith("▁"):
            tok = tok.lstrip("#▁ ")
        if len(tok) < min_len:
            continue
        if tok.lower() in _STOPWORDS:
            continue
        if all(ch.isdigit() for ch in tok):
            continue
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
        if len(out) >= top_k:
            break
    return out


def aggregate_doc_keywords(
    chunk_weights_and_lengths: Iterable[Tuple[Dict[int, float], int]],
    decode: Callable[[List[int]], str],
    top_k: int = 15,
) -> List[str]:
    agg: Dict[int, float] = {}
    for weights, length in chunk_weights_and_lengths:
        weight = float(max(length, 1))
        for tid, val in weights.items():
            agg[tid] = agg.get(tid, 0.0) + val * weight
    return keywords_from_weights(agg, decode=decode, top_k=top_k)
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_embedder_v2.py -v
```

- [ ] **Step 5: Commit**

```bash
git add colab/embedder_v2.py tests/test_embedder_v2.py
git commit -m "feat(embedder-v2): BGE-M3 wrapper + keyword extraction from sparse output"
```

---

## Task 11: Colab HTTP client (list, raw, upsert, progress)

**Files:**
- Create: `colab/client.py`
- Test:   `tests/test_colab_client.py`

- [ ] **Step 1: Failing test with httpx MockTransport**

```python
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
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_colab_client.py -v
```

- [ ] **Step 3: Implement**

```python
# colab/client.py
"""HTTP client used from Colab to talk to the Mac file server."""
from __future__ import annotations

from typing import Optional

import httpx


class MacClient:
    def __init__(self, base_url: str, transport: Optional[httpx.BaseTransport] = None,
                 timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self._client = httpx.Client(transport=transport, timeout=timeout,
                                    http2=True, follow_redirects=True)

    def list_files(self, root: str) -> list[dict]:
        r = self._client.get(f"{self.base}/files/list", params={"root": root})
        r.raise_for_status()
        return r.json()

    def download(self, abs_path: str) -> bytes:
        r = self._client.get(f"{self.base}/files/raw", params={"path": abs_path})
        r.raise_for_status()
        return r.content

    def indexed_state(self, doc_ids: list[str] | None = None) -> dict:
        r = self._client.post(f"{self.base}/admin/indexed-state",
                              json={"doc_ids": doc_ids})
        r.raise_for_status()
        return r.json()

    def delete_doc(self, doc_id: str) -> None:
        r = self._client.delete(f"{self.base}/admin/doc/{doc_id}")
        r.raise_for_status()

    def upsert_chunks_v2(self, points: list[dict]) -> dict:
        r = self._client.post(f"{self.base}/admin/upsert-v2", json={"points": points})
        r.raise_for_status()
        return r.json()

    def push_progress(self, report: dict) -> None:
        r = self._client.post(f"{self.base}/admin/progress", json=report)
        r.raise_for_status()
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_colab_client.py -v
```

- [ ] **Step 5: Commit**

```bash
git add colab/client.py tests/test_colab_client.py
git commit -m "feat(colab-client): httpx wrapper for files + admin endpoints"
```

---

## Task 12: `/admin/upsert-v2` — accept multi-vector points

**Files:**
- Modify: `server/admin_v2.py`
- Test:   `tests/test_admin_v2.py` (extend)

- [ ] **Step 1: Add failing test**

```python
# tests/test_admin_v2.py  (append)
def test_upsert_v2_forwards_to_qdrant(app_and_mock):
    client, qdrant = app_and_mock
    body = {"points": [{
        "id": "abc",
        "dense": [0.0] * 1024,
        "sparse_indices": [1, 2],
        "sparse_values": [0.5, 0.3],
        "colbert_vecs": [[0.0] * 1024],
        "payload": {"doc_id": "d1", "content": "hello"},
    }]}
    r = client.post("/admin/upsert-v2", json=body)
    assert r.status_code == 200
    assert qdrant.upsert.called
    kwargs = qdrant.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "docfinder_v2"
    assert kwargs.get("wait") is True
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_admin_v2.py::test_upsert_v2_forwards_to_qdrant -v
```

- [ ] **Step 3: Implement**

```python
# server/admin_v2.py  (append)
from typing import Any, List
from pydantic import BaseModel


class UpsertPointV2(BaseModel):
    id: str
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    colbert_vecs: List[List[float]]
    payload: dict


class UpsertV2Request(BaseModel):
    points: List[UpsertPointV2]


@router.post("/admin/upsert-v2")
def upsert_v2(body: UpsertV2Request) -> dict:
    q = _require_qdrant()
    points = []
    for p in body.points:
        vectors: dict[str, Any] = {
            "dense": p.dense,
            "colbert": p.colbert_vecs,
            "sparse": qm.SparseVector(indices=p.sparse_indices,
                                      values=p.sparse_values),
        }
        points.append(qm.PointStruct(id=p.id, vector=vectors, payload=p.payload))
    q.upsert(collection_name=_collection, points=points, wait=True)
    return {"upserted": len(points)}
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_admin_v2.py -v
```

- [ ] **Step 5: Commit**

```bash
git add server/admin_v2.py tests/test_admin_v2.py
git commit -m "feat(admin): /admin/upsert-v2 accepts multi-vector points"
```

---

## Task 13: Colab async pipeline + checkpoint

**Files:**
- Create: `colab/pipeline.py`
- Test:   `tests/test_pipeline.py`

- [ ] **Step 1: Failing test (coordinator only, no real GPU)**

```python
# tests/test_pipeline.py
import asyncio
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
import pytest


@pytest.mark.asyncio
async def test_process_one_doc_downloads_extracts_embeds_upserts(tmp_path: Path):
    from colab.pipeline import process_one_doc

    mac = MagicMock()
    mac.download = MagicMock(return_value=b"hello world")
    mac.upsert_chunks_v2 = MagicMock(return_value={"ok": True})

    extractor = MagicMock()
    extractor.return_value = type("R", (), {
        "text": "para one\n\npara two",
        "doc_type": "txt", "page_count": None, "ocr_pages": (),
    })()

    chunker = MagicMock(return_value=[type("C", (), {
        "text": "para one\n\npara two", "token_count": 3, "para_indices": (0, 1),
    })()])

    embedder = MagicMock()
    embedder.encode.return_value = type("E", (), {
        "dense": [[0.0] * 1024],
        "sparse": [([1, 2], [0.5, 0.3])],
        "colbert": [[[0.0] * 1024]],
        "lexical_weights": [{1: 0.5, 2: 0.3}],
    })()

    tokenizer_decode = lambda ids: f"tok{ids[0]}"

    meta = {"path": "a.txt", "abs_path": "/x/a.txt", "doc_id": "d1",
            "size": 10, "mtime": 1, "mode": "full_or_head"}

    await process_one_doc(
        meta, mac_client=mac, extractor=extractor, chunker=chunker,
        embedder=embedder, tokenizer_decode=tokenizer_decode, tmp_dir=tmp_path,
    )

    assert mac.upsert_chunks_v2.called
    point = mac.upsert_chunks_v2.call_args.args[0][0]
    assert point["payload"]["doc_id"] == "d1"
    assert point["payload"]["chunk_idx"] == 0
    assert len(point["dense"]) == 1024


def test_checkpoint_roundtrip(tmp_path: Path):
    from colab.pipeline import Checkpoint
    ck = Checkpoint(tmp_path / "ck.json")
    ck.mark("a", "done")
    ck.mark("b", "failed")
    ck.save()
    ck2 = Checkpoint(tmp_path / "ck.json")
    ck2.load()
    assert ck2.status("a") == "done"
    assert ck2.status("b") == "failed"
    assert ck2.status("c") is None
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Implement**

```python
# colab/pipeline.py
"""Asyncio 3-stage pipeline + checkpoint (spec §9, §11)."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Any

from colab.embedder_v2 import keywords_from_weights, aggregate_doc_keywords
from shared.chunking import chunk_paragraphs, split_paragraphs
from shared.hashing import file_hash_for


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self._state: dict[str, str] = {}

    def load(self) -> None:
        if self.path.exists():
            self._state = json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.write_text(json.dumps(self._state))

    def mark(self, doc_id: str, status: str) -> None:
        self._state[doc_id] = status

    def status(self, doc_id: str) -> str | None:
        return self._state.get(doc_id)


async def process_one_doc(
    meta: dict,
    *,
    mac_client,
    extractor: Callable[..., Any],
    chunker: Callable[..., list],
    embedder,
    tokenizer_decode: Callable[[list[int]], str],
    tmp_dir: Path,
) -> None:
    """Download → extract → chunk → embed → upsert for a single document.

    Atomic flush per doc: upsert_chunks_v2 is called once with all chunks.
    """
    doc_id = meta["doc_id"]
    abs_path = meta["abs_path"]
    mode = meta["mode"]

    raw = mac_client.download(abs_path)
    tmp_path = tmp_dir / f"{doc_id}.bin"
    tmp_path.write_bytes(raw)

    result = extractor(tmp_path, mode=mode)
    paragraphs = split_paragraphs(result.text) or [result.text or meta["path"]]
    chunks = chunker(paragraphs)

    if not chunks:
        return

    texts = [c.text for c in chunks]
    enc = embedder.encode(texts)

    # doc-level keywords aggregated over all chunks
    doc_kws = aggregate_doc_keywords(
        [(enc.lexical_weights[i], chunks[i].token_count) for i in range(len(chunks))],
        decode=tokenizer_decode, top_k=15,
    )

    file_h = file_hash_for(tmp_path)
    points: list[dict] = []
    for i, ch in enumerate(chunks):
        chunk_kws = keywords_from_weights(enc.lexical_weights[i],
                                          decode=tokenizer_decode, top_k=10)
        payload = {
            "doc_id": doc_id,
            "path": meta["path"],
            "abs_path": abs_path,
            "doc_type": result.doc_type,
            "title": Path(meta["path"]).stem,
            "mtime": meta["mtime"],
            "file_hash": file_h,
            "content": ch.text,
            "keywords_chunk": chunk_kws,
            "keywords_doc": doc_kws,
            "page_range": None,
            "chunk_idx": i,
            "chunk_total": len(chunks),
        }
        points.append({
            "id": f"{doc_id}_{i}",
            "dense": enc.dense[i],
            "sparse_indices": enc.sparse[i][0],
            "sparse_values": enc.sparse[i][1],
            "colbert_vecs": enc.colbert[i],
            "payload": payload,
        })

    mac_client.upsert_chunks_v2(points)


async def run_pipeline(
    mac_base_url: str,
    root: str,
    *,
    mac_client,
    extractor,
    embedder,
    tokenizer_decode,
    tmp_dir: Path,
    checkpoint_path: Path,
    http_workers: int = 8,
    progress_every: int = 5,
) -> None:
    ck = Checkpoint(checkpoint_path)
    ck.load()

    files = mac_client.list_files(root=root)
    indexed = mac_client.indexed_state()

    todo: list[dict] = []
    for meta in files:
        prev_mtime = indexed.get(meta["doc_id"])
        if prev_mtime is None:
            todo.append(meta)
        elif meta["mtime"] > prev_mtime:
            mac_client.delete_doc(meta["doc_id"])
            todo.append(meta)

    sem = asyncio.Semaphore(http_workers)
    done_count = 0
    failed_count = 0

    async def worker(meta: dict):
        nonlocal done_count, failed_count
        async with sem:
            try:
                await process_one_doc(
                    meta, mac_client=mac_client, extractor=extractor,
                    chunker=lambda paras: chunk_paragraphs(
                        paras, tokenize_len=lambda t: len(t.split())),
                    embedder=embedder, tokenizer_decode=tokenizer_decode,
                    tmp_dir=tmp_dir,
                )
                ck.mark(meta["doc_id"], "done")
                done_count += 1
            except Exception as e:  # noqa: BLE001
                ck.mark(meta["doc_id"], f"failed:{type(e).__name__}")
                failed_count += 1
            if (done_count + failed_count) % 20 == 0:
                ck.save()
            mac_client.push_progress({
                "total": len(todo),
                "done": done_count,
                "failed": failed_count,
                "current_doc": meta["path"],
                "stage_counts": {"embedded": done_count},
            })

    await asyncio.gather(*(worker(m) for m in todo))
    ck.save()
```

- [ ] **Step 4: Add `pytest-asyncio` to dev-requirements if missing, then run**

```bash
pip install pytest-asyncio
pytest tests/test_pipeline.py -v
```

- [ ] **Step 5: Commit**

```bash
git add colab/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): async 3-stage indexation + atomic flush + checkpoint"
```

---

## Task 14: Search v2 — dense + sparse RRF + ColBERT rerank

**Files:**
- Modify: `server/search.py` (add `search_v2` function, keep old one)
- Test:   `tests/test_search_v2.py`

- [ ] **Step 1: Failing test (mocked Qdrant)**

```python
# tests/test_search_v2.py
from unittest.mock import MagicMock
import numpy as np


def test_search_v2_uses_query_points_with_prefetch():
    from server.search import search_v2
    qdrant = MagicMock()

    class Hit:
        def __init__(self, pid, score, payload):
            self.id, self.score, self.payload = pid, score, payload

    qdrant.query_points.return_value = MagicMock(points=[
        Hit("p1", 0.9, {"doc_id": "d", "content": "hello",
                         "title": "t", "path": "p", "abs_path": "/p",
                         "doc_type": "pdf", "keywords_doc": ["k"]}),
    ])

    embedder = MagicMock()
    embedder.encode.return_value = type("E", (), {
        "dense": [[0.0] * 1024],
        "sparse": [([1, 2], [0.5, 0.3])],
        "colbert": [[[0.0] * 1024, [0.0] * 1024]],
        "lexical_weights": [{1: 0.5, 2: 0.3}],
    })()

    results = search_v2(qdrant, embedder, "my query",
                        collection="docfinder_v2", limit=10)

    kwargs = qdrant.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "docfinder_v2"
    assert kwargs["using"] == "colbert"
    prefetch = kwargs["prefetch"]
    usings = [pf.using for pf in prefetch]
    assert "dense" in usings and "sparse" in usings
    assert kwargs["limit"] == 10
    assert len(results) == 1
    assert results[0].doc_id == "d"
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_search_v2.py -v
```

- [ ] **Step 3: Implement**

```python
# server/search.py  (append — keep existing search() intact)
from qdrant_client.http import models as qm
from shared.schema import SearchResult


def search_v2(qdrant, embedder, query: str,
              collection: str = "docfinder_v2",
              limit: int = 10,
              prefetch_limit: int = 50) -> list[SearchResult]:
    """Dense + sparse with RRF, then ColBERT MaxSim rerank."""
    enc = embedder.encode([query])
    dense_q = enc.dense[0]
    sparse_q = qm.SparseVector(indices=enc.sparse[0][0], values=enc.sparse[0][1])
    colbert_q = enc.colbert[0]

    prefetch = [
        qm.Prefetch(query=dense_q, using="dense", limit=prefetch_limit),
        qm.Prefetch(query=sparse_q, using="sparse", limit=prefetch_limit),
    ]

    resp = qdrant.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=colbert_q,
        using="colbert",
        limit=limit,
        with_payload=True,
    )

    out: list[SearchResult] = []
    for pt in resp.points:
        pl = pt.payload or {}
        out.append(SearchResult(
            chunk_id=str(pt.id),
            doc_id=pl.get("doc_id", ""),
            title=pl.get("title", ""),
            path=pl.get("path", ""),
            abs_path=pl.get("abs_path", ""),
            doc_type=pl.get("doc_type", ""),
            score=float(pt.score),
            excerpt=pl.get("content", ""),
            keywords=pl.get("keywords_doc", []),
        ))
    return out
```

- [ ] **Step 4: Run to verify passing**

```bash
pytest tests/test_search_v2.py -v
```

- [ ] **Step 5: Commit**

```bash
git add server/search.py tests/test_search_v2.py
git commit -m "feat(search-v2): RRF prefetch + ColBERT MaxSim rerank via query_points"
```

---

## Task 15: USE_V2 feature flag in main.py + search endpoint

**Files:**
- Modify: `server/main.py`, `.env.example`

- [ ] **Step 1: Add flag logic with failing integration-style test**

```python
# tests/test_search_endpoint_flag.py
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_search_endpoint_routes_to_v2_when_flag_on(monkeypatch):
    monkeypatch.setenv("USE_V2", "true")
    with patch("server.search.search_v2") as mocked_v2:
        mocked_v2.return_value = []
        from server.main import app  # re-import so env is seen
        client = TestClient(app)
        r = client.post("/search", json={"query": "hello", "limit": 5})
        assert r.status_code == 200
        assert mocked_v2.called


def test_search_endpoint_routes_to_v1_when_flag_off(monkeypatch):
    monkeypatch.setenv("USE_V2", "false")
    with patch("server.main._engine_search") as mocked_v1:
        mocked_v1.return_value = []
        from server.main import app
        client = TestClient(app)
        r = client.post("/search", json={"query": "hello", "limit": 5})
        assert r.status_code == 200
        assert mocked_v1.called
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/test_search_endpoint_flag.py -v
```

- [ ] **Step 3: Implement the dispatch**

In `server/main.py`, at the top of the `/search` handler:

```python
# server/main.py  (inside /search handler)
import os
from server.search import search_v2  # new import

@app.post("/search")
def search_endpoint(body: SearchQuery):
    if os.environ.get("USE_V2", "false").lower() == "true":
        return search_v2(engine.qdrant, engine.embedder_v2, body.query,
                         collection="docfinder_v2", limit=body.limit)
    return _engine_search(body)  # existing path extracted to a helper
```

Wrap existing search body into `_engine_search(body)`.

Add BGE-M3 lazy loader in `engine` (existing SearchEngine class gets an `embedder_v2` attribute, only instantiated when `USE_V2=true`).

- [ ] **Step 4: Update `.env.example`**

```bash
# .env.example
USE_V2=false
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_search_endpoint_flag.py -v
```

- [ ] **Step 6: Commit**

```bash
git add server/main.py .env.example tests/test_search_endpoint_flag.py
git commit -m "feat(search): USE_V2 flag to route /search to v2 engine"
```

---

## Task 16: Admin UI widgets (progress + GPU + current doc)

**Files:**
- Modify: `server/templates/admin.html`

- [ ] **Step 1: Inspect current template to find the "v1" progress block**

```bash
pytest --collect-only tests/ -q >/dev/null  # sanity
grep -n "admin" server/templates/admin.html | head
```

- [ ] **Step 2: Add a v2 section**

Insert above the existing progress block:

```html
<!-- server/templates/admin.html : v2 progress (hidden unless USE_V2) -->
<section id="v2-progress" data-use-v2="{{ use_v2 }}" hidden>
  <h2>Indexation v2 (Colab)</h2>
  <p><strong>Doc courant :</strong> <span id="v2-current-doc">—</span></p>
  <progress id="v2-bar" max="1" value="0"></progress>
  <p><span id="v2-done">0</span> / <span id="v2-total">0</span> docs
     — <span id="v2-rate">0</span> chunks/s — ETA <span id="v2-eta">—</span></p>
  <div class="bars">
    <label>GPU</label>
    <progress id="v2-gpu" max="100" value="0"></progress>
    <label>VRAM</label>
    <progress id="v2-vram" max="16384" value="0"></progress>
  </div>
</section>

<script>
async function pollV2Progress() {
  try {
    const r = await fetch('/admin/progress');
    const p = await r.json();
    document.getElementById('v2-current-doc').textContent = p.current_doc || '—';
    document.getElementById('v2-done').textContent = p.done;
    document.getElementById('v2-total').textContent = p.total;
    document.getElementById('v2-rate').textContent = (p.chunks_per_sec || 0).toFixed(1);
    document.getElementById('v2-eta').textContent = p.eta_seconds + ' s';
    const bar = document.getElementById('v2-bar');
    bar.max = Math.max(1, p.total);
    bar.value = p.done;
    document.getElementById('v2-gpu').value = p.gpu_util_pct || 0;
    document.getElementById('v2-vram').value = p.vram_used_mb || 0;
  } catch (_) {}
}
if (document.getElementById('v2-progress').dataset.useV2 === 'true') {
  document.getElementById('v2-progress').hidden = false;
  setInterval(pollV2Progress, 5000);
  pollV2Progress();
}
</script>
```

Adjust the existing `/admin` handler in `server/main.py` to pass `use_v2=os.environ.get("USE_V2","false")=="true"` to the template.

- [ ] **Step 3: Manual smoke test**

```bash
USE_V2=true uvicorn server.main:app --reload
# open http://127.0.0.1:8000/admin
# verify the v2-progress section renders and polls /admin/progress every 5s
```

- [ ] **Step 4: Commit**

```bash
git add server/templates/admin.html server/main.py
git commit -m "feat(admin-ui): v2 progress widgets (GPU/VRAM/ETA/current doc)"
```

---

## Task 17: Colab notebook wiring — thin wrapper around colab/*

**Files:**
- Modify: `colab_indexer.ipynb` (replace cells) or create `colab_helpers_cell.py`

- [ ] **Step 1: Author a single-cell Python entrypoint**

Create `colab_helpers_cell.py` (a Python script copy-pastable into a Colab cell):

```python
# colab_helpers_cell.py
"""Single-cell Colab entrypoint. Assumes repo was git-cloned to /content/docfinder."""
import asyncio, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, "/content/docfinder")

from colab.client import MacClient
from colab.embedder_v2 import BGEM3Wrapper
from colab.extractor import extract_text, decide_mode
from colab.pipeline import run_pipeline

MAC_BASE = os.environ["MAC_BASE_URL"]   # set to the Cloudflare URL
ROOT = os.environ["DOCFINDER_ROOT"]     # e.g. /Users/julien/Documents

mac = MacClient(MAC_BASE)
embedder = BGEM3Wrapper()
tokenizer_decode = lambda ids: embedder._model_or_build().tokenizer.decode(ids)

tmp = Path(tempfile.mkdtemp())
ck = Path("/content/checkpoint_v2.json")

def extractor(path, mode):
    return extract_text(path, mode=mode)

asyncio.run(run_pipeline(
    MAC_BASE, ROOT,
    mac_client=mac, extractor=extractor, embedder=embedder,
    tokenizer_decode=tokenizer_decode,
    tmp_dir=tmp, checkpoint_path=ck,
))
```

- [ ] **Step 2: Smoke test with a toy root on the Mac**

Create a tiny root with 2 text files, launch the Mac, run the cell locally (not in Colab) pointing `MAC_BASE_URL=http://127.0.0.1:8000`.

```bash
export MAC_BASE_URL=http://127.0.0.1:8000
export DOCFINDER_ROOT=/tmp/mini-docs
mkdir -p /tmp/mini-docs
echo "hello world" > /tmp/mini-docs/a.txt
echo "bonjour le monde" > /tmp/mini-docs/b.txt
USE_V2=true DOCFINDER_ROOTS=/tmp/mini-docs uvicorn server.main:app &
python colab_helpers_cell.py
curl -s http://127.0.0.1:8000/admin/progress | jq
```

Expected: `done=2, total=2, failed=0`.

- [ ] **Step 3: Commit**

```bash
git add colab_helpers_cell.py
git commit -m "feat(colab): single-cell entrypoint wiring pipeline + client + embedder"
```

---

## Task 18: End-to-end dry run on sample corpus

**No new files** — run the full stack against a 20-doc sample.

- [ ] **Step 1: Prepare sample set**

```bash
mkdir -p /tmp/sample-docs
# copy 10 small FR pdfs, 5 EN pdfs, 3 docx, 2 scanned pdfs into /tmp/sample-docs
```

- [ ] **Step 2: Bootstrap Qdrant v2**

```bash
python scripts/setup_qdrant_v2.py
```
Expected output: `docfinder_v2 ready`.

- [ ] **Step 3: Launch Mac in v2 mode**

```bash
USE_V2=true DOCFINDER_ROOTS=/tmp/sample-docs \
  uvicorn server.main:app --host 127.0.0.1 --port 8000 &
```

- [ ] **Step 4: Run pipeline**

```bash
MAC_BASE_URL=http://127.0.0.1:8000 DOCFINDER_ROOT=/tmp/sample-docs \
  python colab_helpers_cell.py
```

- [ ] **Step 5: Verify success criteria (spec §15)**

```bash
curl -s http://127.0.0.1:8000/admin/progress | jq '.done, .failed'
# cross-lingual test:
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"contract renewal","limit":5}' | jq '.[].title'
# keywords sanity check:
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello","limit":1}' | jq '.[].keywords'
```

- [ ] **Step 6: Re-index test (mtime bump)**

```bash
touch /tmp/sample-docs/a.txt
python colab_helpers_cell.py
# expect: only a.txt re-processed (check progress log)
```

- [ ] **Step 7: Commit any fixes + a dry-run note**

```bash
git add -A
git commit -m "chore(v2): end-to-end dry run on 20-doc sample — passes §15 criteria"
```

---

## Task 19: Cleanup old pipeline (post-validation)

**Do NOT run this task until Phase 4 (1-week side-by-side validation) passes.**

**Files to delete:**
- `server/chunks.py` (old /chunks NDJSON)
- `server/indexer.py` (old Mac-side indexer)
- `shared/embedder.py` (e5-large loader)
- `local_indexer.py`
- yake/sentence-transformers/pytesseract lines in `requirements.txt`

- [ ] **Step 1: Verify v2 is live and validated**

```bash
curl -s http://127.0.0.1:8000/search -X POST \
  -H 'Content-Type: application/json' \
  -d '{"query":"anything","limit":1}' | jq '.[].score'
# value should come from search_v2
```

- [ ] **Step 2: Remove files**

```bash
git rm server/chunks.py server/indexer.py shared/embedder.py local_indexer.py
```

- [ ] **Step 3: Prune requirements.txt**

Remove `yake`, `sentence-transformers`, `pytesseract` lines.

- [ ] **Step 4: Remove v1 branch in main.py**

```python
# server/main.py : replace
if os.environ.get("USE_V2"...) == "true":
    return search_v2(...)
return _engine_search(body)
# with simply:
return search_v2(engine.qdrant, engine.embedder_v2, body.query,
                 collection="docfinder_v2", limit=body.limit)
```

- [ ] **Step 5: Delete old collection**

```bash
python -c "from qdrant_client import QdrantClient; \
           QdrantClient('127.0.0.1',6333).delete_collection('docfinder')"
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git commit -m "chore(v2): remove legacy pipeline (e5-large, YAKE, Mac-side indexer)"
```

---

## Post-implementation checklist (spec §15 success criteria)

- [ ] 5 000 docs indexés ≤ 90 min sur T4 (hors download).
- [ ] Mac : aucun process Python > 5 % CPU pendant l'indexation (hors Qdrant + FastAPI).
- [ ] Cross-lingue : requête EN "contract renewal" retrouve un doc FR "renouvellement de contrat" dans top-10.
- [ ] Badges `keywords_doc` propres (pas de dates, pas de stopwords).
- [ ] 1 PDF scanné FR de 10 pages → texte OCR lisible.
- [ ] Après `touch`, seul le doc modifié est ré-indexé.
- [ ] Latence search p95 < 600 ms sur 5 000 docs.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-16-colab-pipeline-v2.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch with checkpoints.

Given user is away (autonomous mode authorized), default: subagent-driven starting at Task 1.
