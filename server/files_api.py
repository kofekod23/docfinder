"""Stateless file server endpoints — Mac only serves bytes and metadata.

Le filtrage (whitelist extensions + blacklist dossiers) est appliqué ici pour
éviter de renvoyer 80k fichiers système. Seuls les documents textuels pertinents
sont exposés à l'indexeur Colab.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from shared.hashing import doc_id_for

router = APIRouter()

FIFTY_MB = 50 * 1024 * 1024

ALLOWED_EXTS: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex",
    ".pdf", ".docx", ".doc", ".odt", ".rtf",
    ".xlsx", ".xls", ".csv", ".tsv", ".ods",
    ".pptx", ".ppt", ".odp", ".key",
    ".xml", ".epub",
    ".pages", ".numbers",
    ".json", ".yaml", ".yml",
})

SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".svn", ".hg", ".bzr",
    "node_modules", "bower_components", "vendor",
    ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox",
    "dist", "build", "target", "out", ".next", ".nuxt", ".turbo",
    ".cache", ".parcel-cache", ".yarn",
    "Library", "Caches", ".Trash", ".Trashes", ".fseventsd",
    ".Spotlight-V100", ".DocumentRevisions-V100",
    ".idea", ".vscode",
    "coverage", "htmlcov",
})


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


def _is_indexable(name: str) -> bool:
    if name.startswith("."):
        return False
    ext = os.path.splitext(name)[1].lower()
    return ext in ALLOWED_EXTS


@router.get("/files/list")
def files_list(root: str = Query(...)) -> List[dict]:
    base = _resolve_under(root)
    out: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for fname in filenames:
            if not _is_indexable(fname):
                continue
            p = Path(dirpath) / fname
            try:
                st = p.stat()
            except OSError:
                continue
            size = st.st_size
            if size == 0:
                continue
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
