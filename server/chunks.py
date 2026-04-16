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

from server.indexer import colab_job_start

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


def _split_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases sur les ponctuations finales."""
    parts = re.split(r"(?<=[.!?»])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _chunk(text: str) -> list[str]:
    """Découpage hybride : paragraphes → phrases → overlap sémantique.

    1. Préserve les sauts de paragraphe (double newline).
    2. Fusionne les petits paragraphes jusqu'à CHUNK_SIZE.
    3. Découpe les paragraphes trop longs par phrase.
    4. Overlap : reprend la queue du chunk précédent depuis un début de phrase.
    """
    # Normalise sans écraser les séparations de paragraphe
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    raw: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > CHUNK_SIZE:
            # Vide le buffer courant avant de découper par phrase
            if current:
                raw.append(current)
                current = ""
            buf = ""
            for sent in _split_sentences(para):
                if len(buf) + len(sent) + 1 <= CHUNK_SIZE:
                    buf = (buf + " " + sent).strip() if buf else sent
                else:
                    if buf:
                        raw.append(buf)
                    buf = sent
            if buf:
                raw.append(buf)
        elif len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                raw.append(current)
            current = para

    if current:
        raw.append(current)

    if not raw:
        return []

    # Overlap sémantique : on reprend la queue du chunk précédent
    # en cherchant un début de phrase pour ne pas couper à mi-mot.
    result: list[str] = [raw[0]]
    for i in range(1, len(raw)):
        tail = raw[i - 1][-CHUNK_OVERLAP:]
        m = re.search(r"(?<=[\.\!\?»])\s+\S", tail)
        if m:
            tail = tail[m.start():].strip()
        elif len(tail) > CHUNK_OVERLAP // 2:
            # Pas de ponctuation trouvée — on prend depuis le premier espace
            sp = tail.find(" ")
            tail = tail[sp + 1:].strip() if sp != -1 else ""
        else:
            tail = ""
        result.append((tail + " " + raw[i]).strip() if tail else raw[i])

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


def _name_hint(file_path: Path) -> str:
    """Construit un texte lisible à partir du nom et du chemin du fichier.

    Remplace les séparateurs (_-) par des espaces pour que YAKE et le modèle
    d'embedding voient des mots normaux plutôt que des tokens collés.
    Ex : "Contrat_Loyer-2024" → "Contrat Loyer 2024"

    Ce hint est préfixé au contenu textuel de chaque fichier pour que le nom
    influence l'embedding dense et les keywords YAKE sparse, même quand le
    texte principal est pauvre ou absent (PDF scanné, fichier chiffré).
    """
    parts = list(file_path.parts)
    cleaned = [re.sub(r"[_\-]+", " ", p) for p in parts]
    return " ".join(cleaned)


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
    hint = _name_hint(file_path)

    if not text.strip():
        # PDF scanné, fichier chiffré, encodage cassé → chemin comme contenu
        text = hint
    else:
        # Préfixe le nom pour que YAKE et l'embedding voient toujours les
        # mots-clés du fichier, même quand le texte principal est générique.
        text = hint + "\n" + text

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


# Nombre de fichiers traités en parallèle côté Mac.
# Chaque worker occupe un thread (extraction texte + YAKE).
# 2 workers suffisent à alimenter le GPU T4 sans surcharger le Mac.
_CHUNK_WORKERS = 4


# Délai max sans émettre de données avant d'envoyer un keepalive (secondes).
# Cloudflare coupe les connexions streaming silencieuses après ~30s.
_KEEPALIVE_INTERVAL = 10.0

_KEEPALIVE_LINE = json.dumps({"type": "keepalive"}).encode() + b"\n"


async def iter_chunks_json(path: str) -> AsyncGenerator[bytes, None]:
    """
    Génère des lignes NDJSON, une par chunk.

    Chaque opération bloquante (scan disque, extraction texte, YAKE) tourne
    dans un thread pool via run_in_executor pour ne jamais bloquer l'event loop.
    _CHUNK_WORKERS fichiers sont traités en parallèle pour saturer le GPU Colab.

    Un keepalive est émis toutes les _KEEPALIVE_INTERVAL secondes sans données
    pour éviter que Cloudflare (ou tout proxy) coupe la connexion streaming.
    """
    loop = asyncio.get_event_loop()
    root = Path(path).expanduser()

    if not root.exists():
        yield json.dumps({"error": f"Dossier introuvable : {path}"}).encode() + b"\n"
        return

    # Scan disque dans un thread — peut être lent sur iCloud
    files: list[Path] = await loop.run_in_executor(None, _iter_files, root)

    yield json.dumps({"type": "meta", "total_files": len(files)}).encode() + b"\n"

    # Signaler le démarrage au job tracker pour que l'UI reflète la progression Colab
    colab_job_start(path, len(files))

    sem = asyncio.Semaphore(_CHUNK_WORKERS)

    async def _process(fp: Path) -> dict:
        async with sem:
            return await loop.run_in_executor(None, _process_file, fp, root)

    pending: set[asyncio.Task] = {asyncio.ensure_future(_process(fp)) for fp in files}

    last_emit = loop.time()

    while pending:
        done_set, pending = await asyncio.wait(
            pending,
            timeout=_KEEPALIVE_INTERVAL,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done_set:
            # Timeout sans résultat — envoyer un keepalive
            yield _KEEPALIVE_LINE
            last_emit = loop.time()
            continue

        for task in done_set:
            result: dict = task.result()
            last_emit = loop.time()

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
