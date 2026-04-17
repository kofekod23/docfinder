"""V2 admin endpoints (spec §11-§13)."""
from __future__ import annotations

import uuid
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

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
        try:
            points, offset = q.scroll(
                collection_name=_collection,
                limit=10_000,
                with_payload=["doc_id", "mtime"],
                with_vectors=False,
                offset=offset,
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return {}
            raise
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


class UpsertPointV2(BaseModel):
    id: str
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    colbert_vecs: List[List[float]]
    payload: dict


class UpsertV2Request(BaseModel):
    points: List[UpsertPointV2]


# Qdrant default JSON body limit is 32 MiB. ColBERT multi-vectors dominate
# payload size (~20 bytes per float in JSON), so we sub-batch by estimated
# bytes rather than by point count.
_UPSERT_SOFT_LIMIT_BYTES = 20 * 1024 * 1024


def _estimate_point_bytes(p: UpsertPointV2) -> int:
    colbert_floats = sum(len(v) for v in p.colbert_vecs)
    return (len(p.dense) + colbert_floats + len(p.sparse_values)) * 20


@router.post("/admin/upsert-v2")
def upsert_v2(body: UpsertV2Request) -> dict:
    q = _require_qdrant()
    batch: list = []
    batch_bytes = 0
    upserted = 0

    def flush() -> None:
        nonlocal batch, batch_bytes, upserted
        if not batch:
            return
        q.upsert(collection_name=_collection, points=batch, wait=True)
        upserted += len(batch)
        batch = []
        batch_bytes = 0

    for p in body.points:
        vectors: dict[str, Any] = {
            "dense": p.dense,
            "colbert": p.colbert_vecs,
            "sparse": qm.SparseVector(indices=p.sparse_indices,
                                      values=p.sparse_values),
        }
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, p.id))
        point = qm.PointStruct(id=pid, vector=vectors, payload=p.payload)
        est = _estimate_point_bytes(p)
        if batch and batch_bytes + est > _UPSERT_SOFT_LIMIT_BYTES:
            flush()
        batch.append(point)
        batch_bytes += est
    flush()
    return {"upserted": upserted}
