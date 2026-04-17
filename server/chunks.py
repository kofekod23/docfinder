"""
Générateur de blocs texte pour l'endpoint /chunks.

Extrait le texte des documents et émet une ligne JSON par fichier (NDJSON).
Colab consomme ce flux pour effectuer le chunking sémantique, le calcul des
vecteurs sparse YAKE et l'embedding dense sur GPU.

PDF scannés : fallback OCR via pytesseract si la page renvoie < 10 caractères.

Toutes les opérations bloquantes (I/O fichier, extraction PDF/DOCX, OCR)
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

from server.indexer import colab_job_start, load_indexed_hashes

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}


def _file_hash(path: Path) -> str:
    """Empreinte rapide basée sur mtime + taille — aucune lecture du contenu."""
    try:
        st = path.stat()
        return f"{st.st_mtime:.3f}_{st.st_size}"
    except OSError:
        return ""


def _extract_text(path: Path) -> str:
    """Extrait le texte brut d'un fichier (bloquant — appeler via run_in_executor).

    PDF scannés : si une page retourne < 10 caractères, tente un OCR via
    pytesseract (résolution ×2 pour une meilleure qualité). Les pages sans texte
    ni OCR sont ignorées silencieusement.

    Parseurs : pymupdf (.pdf), python-docx (.docx/.doc), lecture directe (.txt/.md).
    Retourne "" si le fichier est illisible (PDF chiffré, encodage cassé).
    """
    try:
        s = path.suffix.lower()
        if s == ".pdf":
            import fitz
            page_texts: list[str] = []
            for page in fitz.open(str(path)):
                text = page.get_text()
                if len(text.strip()) < 10:
                    try:
                        import io

                        import pytesseract
                        from PIL import Image

                        mat = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=mat)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = pytesseract.image_to_string(img, lang="fra+eng")
                    except Exception as ocr_err:
                        log.warning(
                            "OCR échoué page %d de %s : %s",
                            page.number, path.name, ocr_err,
                        )
                page_texts.append(text)
            return "\n".join(page_texts)
        if s in {".docx", ".doc"}:
            from docx import Document
            return "\n".join(p.text for p in Document(str(path)).paragraphs)
        if s in {".txt", ".md"}:
            with path.open(errors="ignore") as f:
                return f.read(1024 * 1024)  # 1 Mo max
    except Exception as e:
        log.warning("Extraction échouée %s : %s", path.name, e)
    return ""


def _iter_files(
    root: Path, exclude: frozenset[str] | None = None
) -> list[Path]:
    """Retourne la liste complète des fichiers indexables sous root (bloquant).

    exclude : ensemble de noms de sous-répertoires immédiats à ignorer.
    Exclusions systématiques : .icloud, ~$*, .* (cachés macOS).
    """
    result = []
    for p in root.rglob("*"):
        if p.suffix == ".icloud":
            continue
        if p.name.startswith("~$") or p.name.startswith("."):
            continue
        if exclude:
            try:
                rel = p.relative_to(root)
                if rel.parts and rel.parts[0] in exclude:
                    continue
            except ValueError:
                pass
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            result.append(p)
    return result


def _name_hint(file_path: Path) -> str:
    """Construit un texte lisible à partir du nom et du chemin du fichier.

    Remplace les séparateurs (_-) par des espaces pour que le modèle d'embedding
    voie des mots normaux plutôt que des tokens collés.
    Ex : "Contrat_Loyer-2024" → "Contrat Loyer 2024"
    """
    parts = list(file_path.parts)
    cleaned = [re.sub(r"[_\-]+", " ", p) for p in parts]
    return " ".join(cleaned)


def _process_file(
    file_path: Path, root: Path, existing_hashes: dict[str, str]
) -> dict:
    """Traite un fichier et retourne son bloc de texte brut (bloquant).

    Si le fichier est déjà indexé (même file_hash), retourne skip=True.
    Le chunking sémantique, YAKE et l'embedding sont délégués à Colab GPU.
    """
    rel = str(file_path.relative_to(root))
    doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
    fhash = _file_hash(file_path)
    mtime = int(file_path.stat().st_mtime) if fhash else 0
    if fhash and existing_hashes.get(doc_id) == fhash:
        return {"skip": True, "rel": rel}

    suffix = file_path.suffix.lower()
    doc_type = (
        "pdf"  if suffix == ".pdf"            else
        "docx" if suffix in {".docx", ".doc"} else
        "txt"  if suffix == ".txt"            else "md"
    )

    text = _extract_text(file_path)
    hint = _name_hint(file_path)

    if not text.strip():
        text = hint
    else:
        text = hint + "\n" + text

    if not text.strip():
        return {"skip": True, "rel": rel}

    return {
        "skip": False,
        "rel": rel,
        "block": {
            "type":      "block",
            "doc_id":    doc_id,
            "title":     file_path.stem,
            "path":      rel,
            "abs_path":  str(file_path),
            "doc_type":  doc_type,
            "content":   text,
            "file_hash": fhash,
            "mtime":     mtime,
        },
    }


# Nombre de fichiers traités en parallèle côté Mac.
# Chaque worker occupe un thread (extraction texte, OCR éventuel).
_CHUNK_WORKERS = 4

# Délai max sans émettre de données avant d'envoyer un keepalive (secondes).
# Cloudflare coupe les connexions streaming silencieuses après ~30s.
_KEEPALIVE_INTERVAL = 10.0

_KEEPALIVE_LINE = json.dumps({"type": "keepalive"}).encode() + b"\n"


async def iter_chunks_json(
    path: str, exclude: list[str] | None = None
) -> AsyncGenerator[bytes, None]:
    """
    Génère des lignes NDJSON, une par fichier.

    Protocole :
      {"type": "meta",  "total_files": N}          — nb total de fichiers
      {"type": "file",  "path": "..."}              — fichier en cours
      {"type": "block", ...}                        — texte brut du fichier
      {"type": "skip",  "path": "..."}              — fichier déjà indexé
      {"type": "keepalive"}                         — anti-timeout Cloudflare
      {"type": "done"}                              — fin du flux

    Chaque opération bloquante (scan disque, extraction texte, OCR) tourne
    dans un thread pool via run_in_executor pour ne jamais bloquer l'event loop.
    """
    loop = asyncio.get_event_loop()
    root = Path(path).expanduser()

    if not root.exists():
        yield json.dumps({"error": f"Dossier introuvable : {path}"}).encode() + b"\n"
        return

    excl = frozenset(exclude) if exclude else None

    files, existing_hashes = await asyncio.gather(
        loop.run_in_executor(None, _iter_files, root, excl),
        loop.run_in_executor(None, load_indexed_hashes),
    )

    yield json.dumps({"type": "meta", "total_files": len(files)}).encode() + b"\n"

    colab_job_start(path, len(files))

    sem = asyncio.Semaphore(_CHUNK_WORKERS)

    async def _process(fp: Path) -> dict:
        async with sem:
            return await loop.run_in_executor(None, _process_file, fp, root, existing_hashes)

    pending: set[asyncio.Task] = {asyncio.ensure_future(_process(fp)) for fp in files}

    while pending:
        done_set, pending = await asyncio.wait(
            pending,
            timeout=_KEEPALIVE_INTERVAL,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done_set:
            yield _KEEPALIVE_LINE
            continue

        for task in done_set:
            result: dict = task.result()

            if result["skip"]:
                yield json.dumps({"type": "skip", "path": result["rel"]}).encode() + b"\n"
                continue

            yield json.dumps({"type": "file", "path": result["rel"]}).encode() + b"\n"
            yield json.dumps(result["block"]).encode() + b"\n"

    yield json.dumps({"type": "done"}).encode() + b"\n"
