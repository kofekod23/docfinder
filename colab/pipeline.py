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
