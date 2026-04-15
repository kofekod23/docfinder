"""
Tâche d'indexation en arrière-plan.

Gère un seul job à la fois via un thread daemon.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

import yake
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from shared.embedder import Embedder

log = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
COLLECTION = "docfinder"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BATCH_SIZE = 32
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
ICLOUD_DEFAULT = str(Path.home() / "Library/Mobile Documents/com~apple~CloudDocs")

_yake_extractor = yake.KeywordExtractor(lan="fr", n=3, dedupLim=0.7, top=20)


class JobStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class IndexJob:
    status: JobStatus = JobStatus.IDLE
    path: str = ""
    total: int = 0
    done: int = 0
    skipped: int = 0
    chunks: int = 0
    current_file: str = ""
    error: str = ""
    log_lines: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "path": self.path,
            "total": self.total,
            "done": self.done,
            "skipped": self.skipped,
            "chunks": self.chunks,
            "current_file": self.current_file,
            "error": self.error,
            "log": self.log_lines[-50:],  # dernières 50 lignes
            "progress_pct": round(self.done / self.total * 100) if self.total else 0,
        }


# Job courant (un seul à la fois)
_job: IndexJob = IndexJob()
_lock = threading.Lock()
_cancel_event = threading.Event()


def current_job() -> dict:
    with _lock:
        return _job.to_dict()


def upsert_points(points_data: list[dict]) -> int:
    """Insère des points (envoyés par Colab) dans Qdrant local."""
    client = QdrantClient(url=QDRANT_URL)
    structs: list[PointStruct] = []
    for p in points_data:
        vector_dict: dict = {"dense": p["dense"]}
        if p.get("sparse_indices"):
            vector_dict["sparse"] = SparseVector(
                indices=p["sparse_indices"],
                values=p["sparse_values"],
            )
        structs.append(PointStruct(
            id=p["id"],
            vector=vector_dict,
            payload=p["payload"],
        ))
    for i in range(0, len(structs), BATCH_SIZE):
        client.upsert(collection_name=COLLECTION, points=structs[i : i + BATCH_SIZE], wait=False)
    return len(structs)


def start_indexation(path: str, reset: bool = False) -> dict:
    """Lance l'indexation dans un thread daemon. Retourne une erreur si déjà en cours."""
    global _job
    with _lock:
        if _job.status == JobStatus.RUNNING:
            return {"error": "Une indexation est déjà en cours."}
        _job = IndexJob(status=JobStatus.RUNNING, path=path)
    _cancel_event.clear()

    t = threading.Thread(target=_run, args=(path, reset), daemon=True)
    t.start()
    return {"started": True, "path": path}


def cancel_indexation() -> dict:
    """Demande l'annulation du job en cours."""
    with _lock:
        if _job.status != JobStatus.RUNNING:
            return {"error": "Aucune indexation en cours."}
    _cancel_event.set()
    return {"cancelled": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    log.info(msg)
    with _lock:
        _job.log_lines.append(msg)


def _kw_index(kw: str) -> int:
    return int(hashlib.md5(kw.lower().encode()).hexdigest(), 16) % (2 ** 20)


def _build_sparse(text: str) -> tuple[list[int], list[float]]:
    pairs = _yake_extractor.extract_keywords(text)
    if not pairs:
        return [], []
    raw = {_kw_index(kw): 1.0 / (s + 1e-9) for kw, s in pairs}
    max_v = max(raw.values())
    return list(raw.keys()), [v / max_v for v in raw.values()]


def _extract_text(path: Path) -> str:
    try:
        s = path.suffix.lower()
        if s == ".pdf":
            import fitz
            return "\n".join(p.get_text() for p in fitz.open(str(path)))
        if s in {".docx", ".doc"}:
            from docx import Document
            return "\n".join(p.text for p in Document(str(path)).paragraphs)
        if s in {".txt", ".md"}:
            return path.read_text(errors="ignore")
    except Exception as e:
        _log(f"  ⚠ extraction échouée : {e}")
    return ""


def _chunk(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    result, start = [], 0
    while start < len(text):
        result.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return result


def _iter_files(root: Path):
    for p in root.rglob("*"):
        if p.suffix == ".icloud":
            continue
        if p.name.startswith("~$") or p.name.startswith("."):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run(path: str, reset: bool) -> None:
    global _job
    try:
        root = Path(path).expanduser()
        if not root.exists():
            with _lock:
                _job.status = JobStatus.ERROR
                _job.error = f"Dossier introuvable : {path}"
            return

        client = QdrantClient(url=QDRANT_URL)

        if reset:
            _log("Réinitialisation de la collection…")
            try:
                client.delete_collection(COLLECTION)
            except Exception:
                pass
            import subprocess, sys
            subprocess.run([sys.executable, "setup_qdrant.py"], check=True, capture_output=True)
            _log("Collection recréée.")

        files = list(_iter_files(root))
        with _lock:
            _job.total = len(files)
        _log(f"{len(files)} fichiers trouvés dans {root}")

        embedder = Embedder.get_instance()

        for file_path in files:
            if _cancel_event.is_set():
                _log("Indexation annulée.")
                with _lock:
                    _job.status = JobStatus.ERROR
                    _job.error = "Annulée par l'utilisateur."
                    _job.current_file = ""
                return

            rel = str(file_path.relative_to(root))
            with _lock:
                _job.current_file = rel

            text = _extract_text(file_path)
            if not text.strip():
                # Texte vide (PDF scanné, fichier chiffré…) → chemin complet comme contenu
                parts = [p.replace("_", " ").replace("-", " ") for p in file_path.parts]
                text = " ".join(parts)
                _log(f"  {rel} → texte vide, chemin utilisé comme contenu")

            chunks = _chunk(text)
            _log(f"  {rel} → {len(chunks)} chunks")

            doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
            suffix = file_path.suffix.lower()
            doc_type = (
                "pdf"  if suffix == ".pdf"          else
                "docx" if suffix in {".docx", ".doc"} else
                "txt"  if suffix == ".txt"           else "md"
            )

            embeddings = embedder.encode_batch(chunks, batch_size=BATCH_SIZE)
            points: list[PointStruct] = []

            for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_{idx}"
                kw_pairs = _yake_extractor.extract_keywords(chunk_text)
                keywords = [kw for kw, _ in kw_pairs]
                sp_i, sp_v = _build_sparse(chunk_text)

                points.append(PointStruct(
                    id=int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2 ** 63),
                    vector={
                        "dense": emb.tolist(),
                        "sparse": SparseVector(indices=sp_i, values=sp_v),
                    },
                    payload={
                        "doc_id": doc_id, "chunk_id": chunk_id,
                        "title": file_path.stem, "path": rel,
                        "abs_path": str(file_path),
                        "doc_type": doc_type, "content": chunk_text,
                        "keywords": keywords, "chunk_index": idx,
                    },
                ))

            for i in range(0, len(points), BATCH_SIZE):
                client.upsert(collection_name=COLLECTION, points=points[i: i + BATCH_SIZE])

            with _lock:
                _job.chunks += len(points)
                _job.done += 1

        _log(f"Terminé — {_job.chunks} chunks insérés, {_job.skipped} fichiers ignorés.")
        with _lock:
            _job.status = JobStatus.DONE
            _job.current_file = ""

    except Exception as e:
        log.exception("Erreur indexation")
        with _lock:
            _job.status = JobStatus.ERROR
            _job.error = str(e)
