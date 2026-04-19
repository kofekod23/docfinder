"""Stateless file server endpoints — Mac only serves bytes and metadata.

Le filtrage (whitelist extensions + blacklist dossiers) est appliqué ici pour
éviter de renvoyer 80k fichiers système. Seuls les documents textuels pertinents
sont exposés à l'indexeur Colab.
"""
from __future__ import annotations

import errno
import os
import threading
import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from shared.hashing import doc_id_for

# Sérialisation des reads disque : macOS + File Provider Extension (Google Drive
# FS / iCloud) renvoie EDEADLK sur reads concurrents dans le même process uvicorn.
# Un seul read à la fois élimine le problème ; l'overhead sur un throughput typique
# (Colab 16 workers parallèles, fichiers ~1 MB) reste négligeable car le lock
# n'est tenu que pendant la lecture disque (quelques ms).
_FILE_READ_LOCK = threading.Lock()

router = APIRouter()

FIFTY_MB = 50 * 1024 * 1024

ALLOWED_EXTS: frozenset[str] = frozenset({
    ".pdf",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".xlsx", ".xls",
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

    # Lecture via subprocess `cat` pour contourner EDEADLK sur fichiers Google
    # Drive FS "online only" (le in-process open déclenche un deadlock sous
    # uvicorn threadpool — diagnostic 2026-04-19). Pré-warm via `head -c 1`
    # force le File Provider à matérialiser le fichier avant le cat complet ;
    # sans ce pré-warm, `cat` direct lui-même échoue en EDEADLK. Avec pré-warm,
    # la probabilité de succès monte à 100% sur les fichiers testés.
    # Fichiers > 50 MB filtrés côté listing donc RAM safe.
    import subprocess
    data: bytes | None = None
    last_err: str = ""
    for attempt in range(6):
        try:
            with _FILE_READ_LOCK:
                # Pré-warm matérialisation Google Drive FS (best-effort).
                subprocess.run(
                    ["head", "-c", "1", str(p)],
                    capture_output=True,
                    timeout=30,
                )
                # Lecture complète.
                result = subprocess.run(
                    ["cat", str(p)],
                    capture_output=True,
                    timeout=120,
                )
        except subprocess.TimeoutExpired:
            raise HTTPException(504, f"read timeout for {p.name}")
        if result.returncode == 0:
            data = result.stdout
            break
        last_err = result.stderr.decode("utf-8", "replace")[:300]
        if "deadlock" in last_err.lower() and attempt < 5:
            time.sleep(1.0 * (2 ** attempt))  # 1, 2, 4, 8, 16s
            continue
        # Autre erreur → abandon immédiat
        raise HTTPException(
            500,
            f"cat failed (rc={result.returncode}): {last_err}",
        )
    if data is None:
        # Tous les essais ont échoué sur deadlock → 503 retryable côté Colab.
        raise HTTPException(
            503,
            f"persistent deadlock after 6 retries: {last_err}",
        )

    def stream():
        yield data

    return StreamingResponse(stream(), media_type="application/octet-stream")
