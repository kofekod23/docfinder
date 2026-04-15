"""
Générateur de chunks pour l'endpoint /chunks.

Extrait le texte des documents, les découpe, calcule les sparse vectors YAKE,
et émet une ligne JSON par chunk (NDJSON).
Colab consomme ce flux pour calculer les embeddings sur GPU.

Toutes les opérations bloquantes (I/O fichier, extraction PDF/DOCX, YAKE)
tournent dans un thread pool via run_in_executor pour ne pas bloquer
l'event loop asyncio — ce qui garantit un vrai streaming HTTP incrémental.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator

import yake

log = logging.getLogger(__name__)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

_yake_extractor = yake.KeywordExtractor(lan="fr", n=3, dedupLim=0.7, top=20)


def _kw_index(kw: str) -> int:
    """Hash MD5 → indice entier dans l'espace sparse [0, 2^20).

    Identique à server/indexer.py et local_indexer.py — tous les modules
    doivent rester synchronisés pour que les vecteurs sparse soient comparables.
    """
    return int(hashlib.md5(kw.lower().encode()).hexdigest(), 16) % (2 ** 20)


def _build_sparse(text: str) -> tuple[list[int], list[float]]:
    """Vecteur sparse YAKE : n-grammes → (indices, values) normalisés dans [0, 1].

    Utilisé ici pour pré-calculer les sparse vectors côté serveur local avant
    d'envoyer les chunks à Colab. Colab n'a donc qu'à ajouter les embeddings dense.
    """
    pairs = _yake_extractor.extract_keywords(text)
    if not pairs:
        return [], []
    raw = {_kw_index(kw): 1.0 / (s + 1e-9) for kw, s in pairs}
    max_v = max(raw.values())
    return list(raw.keys()), [v / max_v for v in raw.values()]


def _extract_text(path: Path) -> str:
    """Extrait le texte brut d'un fichier (bloquant — appeler via run_in_executor).

    Parseurs : pymupdf (.pdf), python-docx (.docx/.doc), lecture directe (.txt/.md).
    Retourne "" si le fichier est illisible (PDF scanné, chiffré, encodage cassé).
    """
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
        log.warning("Extraction échouée %s : %s", path.name, e)
    return ""


def _chunk(text: str) -> list[str]:
    """Découpe le texte normalisé en chunks de 1500 chars avec overlap 200.

    Normalise d'abord les espaces/retours à la ligne en espace simple.
    Même paramètres que server/indexer.py pour garantir la cohérence des points Qdrant.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    result, start = [], 0
    while start < len(text):
        result.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return result


def _iter_files(root: Path) -> list[Path]:
    """Retourne la liste complète des fichiers indexables sous root (bloquant).

    Contrairement à server/indexer.py qui yield, retourne une liste complète
    car le total est émis dans la ligne NDJSON {type: "meta"} avant le streaming.
    Exclusions identiques : .icloud, ~$*, .* (cachés macOS).
    """
    result = []
    for p in root.rglob("*"):
        if p.suffix == ".icloud":
            continue
        if p.name.startswith("~$") or p.name.startswith("."):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            result.append(p)
    return result


def _process_file(file_path: Path, root: Path) -> dict:
    """Traite un fichier et retourne ses chunks + métadonnées (bloquant)."""
    rel = str(file_path.relative_to(root))
    doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
    suffix = file_path.suffix.lower()
    doc_type = (
        "pdf"  if suffix == ".pdf"            else
        "docx" if suffix in {".docx", ".doc"} else
        "txt"  if suffix == ".txt"            else "md"
    )

    text = _extract_text(file_path)
    chunks = _chunk(text)

    if not chunks:
        return {"skip": True, "rel": rel}

    chunk_lines = []
    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_{idx}"
        kw_pairs = _yake_extractor.extract_keywords(chunk_text)
        keywords = [kw for kw, _ in kw_pairs]
        sp_i, sp_v = _build_sparse(chunk_text)
        chunk_lines.append({
            "type": "chunk",
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "point_id": int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2 ** 63),
            "title": file_path.stem,
            "path": rel,
            "abs_path": str(file_path),
            "doc_type": doc_type,
            "content": chunk_text,
            "keywords": keywords,
            "chunk_index": idx,
            "sparse_indices": sp_i,
            "sparse_values": sp_v,
        })

    return {"skip": False, "rel": rel, "chunks": chunk_lines}


async def iter_chunks_json(path: str) -> AsyncGenerator[bytes, None]:
    """
    Génère des lignes NDJSON, une par chunk.

    Chaque opération bloquante (scan disque, extraction texte, YAKE) tourne
    dans un thread pool via run_in_executor pour ne jamais bloquer l'event loop.
    Cela garantit que FastAPI envoie chaque ligne dès qu'elle est prête,
    sans bufferiser l'intégralité de la réponse.
    """
    loop = asyncio.get_event_loop()
    root = Path(path).expanduser()

    if not root.exists():
        yield json.dumps({"error": f"Dossier introuvable : {path}"}).encode() + b"\n"
        return

    # Scan disque dans un thread — peut être lent sur iCloud
    files: list[Path] = await loop.run_in_executor(None, _iter_files, root)

    yield json.dumps({"type": "meta", "total_files": len(files)}).encode() + b"\n"

    for file_path in files:
        # Extraction + chunking + YAKE dans un thread
        result: dict = await loop.run_in_executor(None, _process_file, file_path, root)

        if result["skip"]:
            yield json.dumps({"type": "skip", "path": result["rel"]}).encode() + b"\n"
            continue

        yield json.dumps({
            "type": "file",
            "path": result["rel"],
            "chunks": len(result["chunks"]),
        }).encode() + b"\n"

        for chunk_data in result["chunks"]:
            yield json.dumps(chunk_data).encode() + b"\n"

    yield json.dumps({"type": "done"}).encode() + b"\n"
