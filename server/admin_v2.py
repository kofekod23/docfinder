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
