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
