"""
Générateur de chunks pour l'endpoint /chunks.

Extrait le texte des documents, les découpe en fenêtres de 1500 chars,
et émet une ligne JSON par chunk (NDJSON). Le serveur n'émet que du texte brut :
Colab calcule YAKE + sparse + dense côté GPU/CPU Colab (D15).

Toutes les opérations bloquantes (I/O fichier, extraction PDF/DOCX) tournent
dans un thread pool via run_in_executor pour ne pas bloquer l'event loop asyncio
— ce qui garantit un vrai streaming HTTP incrémental.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator

log = logging.getLogger(__name__)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}


def _extract_text(path: Path) -> str:
    """Extrait le texte brut d'un fichier (bloquant — appeler via run_in_executor).

    Parseurs : pymupdf (.pdf), python-docx (.docx/.doc), lecture directe (.txt/.md).
    Retourne "" si le fichier est illisible (PDF scanné, chiffré, encodage cassé).

    Utilise un `with fitz.open()` : garantit la libération du handle et de la RAM
    du PDF même en cas d'exception — critique sur les gros corpus (D15).
    """
    try:
        s = path.suffix.lower()
        if s == ".pdf":
            import fitz
            with fitz.open(str(path)) as doc:
                return "\n".join(p.get_text() for p in doc)
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
    """Traite un fichier et retourne ses chunks bruts + métadonnées (bloquant).

    N'effectue PAS le calcul YAKE/sparse/keywords — Colab le fait (D15).
    Le Mac se concentre sur l'I/O et l'extraction texte ; tout le CPU-heavy
    (YAKE pur Python) est déporté sur Colab qui a du cycle à revendre.
    """
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
            "chunk_index": idx,
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

    for idx, file_path in enumerate(files):
        # Extraction + chunking dans un thread (YAKE est fait côté Colab — D15)
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

        # Libérer la RAM agressivement entre fichiers — sur un Mac âgé,
        # pymupdf peut retenir plusieurs dizaines de MB par gros PDF.
        if idx % 10 == 9:
            gc.collect()

    yield json.dumps({"type": "done"}).encode() + b"\n"
