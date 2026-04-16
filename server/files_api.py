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
