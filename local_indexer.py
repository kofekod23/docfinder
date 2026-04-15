"""
Indexeur local CLI — documents iCloud → Qdrant (local_indexer.py).

Script autonome (pas de serveur FastAPI) pour indexer un dossier entier en une passe.
Contrairement à server/indexer.py (thread daemon, invoqué via HTTP), ce script
s'exécute directement en ligne de commande et bloque jusqu'à la fin.

Flux de données :
    fichiers disque → extract_text() → chunk_text() → build_sparse() (YAKE)
    → Embedder.encode_batch() (CPU) → PointStruct → client.upsert() (Qdrant local)

Différences clés avec server/indexer.py :
- Script CLI autonome (pas importé par main.py)
- Pas de rapport de progression HTTP — logs console uniquement
- Pas de support de l'annulation (pas de cancel_event)
- N'insère PAS abs_path dans le payload (contrairement à server/indexer.py — legacy)

Usage :
    python local_indexer.py                          # indexe ~/Library/Mobile Documents/com~apple~CloudDocs/
    python local_indexer.py --path /autre/dossier    # dossier personnalisé
    python local_indexer.py --reset                  # vide la collection avant d'indexer
    python local_indexer.py --dry-run                # liste les fichiers sans indexer
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Generator

import yake
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from shared.embedder import Embedder
from shared.schema import DocumentChunk, DocType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ICLOUD_PATH = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"
QDRANT_URL  = "http://localhost:6333"
COLLECTION  = "docfinder"
CHUNK_SIZE  = 1500
CHUNK_OVERLAP = 200
BATCH_SIZE  = 32
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAKE sparse vectors
# ---------------------------------------------------------------------------
_yake = yake.KeywordExtractor(lan="fr", n=3, dedupLim=0.7, top=20)


def _kw_to_index(kw: str) -> int:
    """Hash MD5 d'un mot-clé → indice entier dans l'espace sparse [0, 2^20).

    Identique à server/indexer.py._kw_index() et server/chunks.py._kw_index().
    Les trois modules doivent rester synchronisés pour la cohérence des vecteurs.
    """
    return int(hashlib.md5(kw.lower().encode()).hexdigest(), 16) % (2**20)


def build_sparse(text: str) -> tuple[list[int], list[float]]:
    """Vecteur sparse YAKE normalisé — même algorithme que server/indexer.py._build_sparse().

    Retourne (indices, values) prêts pour qdrant_client.models.SparseVector.
    Score YAKE inversé (1/score) puis normalisé dans [0, 1] par rapport au max.
    """
    pairs = _yake.extract_keywords(text)
    if not pairs:
        return [], []
    raw = {_kw_to_index(kw): 1.0 / (score + 1e-9) for kw, score in pairs}
    max_v = max(raw.values())
    indices = list(raw.keys())
    values  = [v / max_v for v in raw.values()]
    return indices, values


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_text(path: Path) -> str:
    """Extrait le texte brut d'un fichier selon son extension.

    Parseurs : pymupdf (PDF), python-docx (DOCX/DOC), lecture directe (TXT/MD).
    Contrairement à server/indexer.py, les fichiers à texte vide sont ignorés (skipped)
    plutôt que remplacés par le chemin — comportement legacy CLI.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            return "\n".join(page.get_text() for page in doc)
        elif suffix in {".docx", ".doc"}:
            from docx import Document
            return "\n".join(p.text for p in Document(str(path)).paragraphs)
        elif suffix in {".txt", ".md"}:
            return path.read_text(errors="ignore")
    except Exception as e:
        log.warning("Extraction échouée %s : %s", path.name, e)
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe le texte en chunks de `size` chars avec un overlap de `overlap` chars.

    Normalise d'abord les whitespace. Paramètres par défaut identiques à server/indexer.py
    (1500 chars, overlap 200) pour garantir la cohérence des points Qdrant (D7).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------
def iter_documents(root: Path) -> Generator[Path, None, None]:
    """Générateur récursif — yield les fichiers indexables sous root.

    Exclusions :
    - .icloud  : placeholders iCloud non téléchargés (fichier absent en local)
    - ~$*      : fichiers temporaires Office (document ouvert dans Word/Excel)
    - .*       : fichiers/dossiers cachés macOS (.DS_Store, .Spotlight, .git…)
    """
    for path in root.rglob("*"):
        if path.suffix == ".icloud":
            continue
        if path.name.startswith("~$") or path.name.startswith("."):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Indexeur local DocFinder")
    parser.add_argument("--path",    default=str(ICLOUD_PATH), help="Dossier racine à indexer")
    parser.add_argument("--reset",   action="store_true", help="Vider la collection avant d'indexer")
    parser.add_argument("--dry-run", action="store_true", help="Lister les fichiers sans indexer")
    args = parser.parse_args()

    root = Path(args.path).expanduser()
    if not root.exists():
        log.error("Dossier introuvable : %s", root)
        sys.exit(1)

    # Qdrant
    client = QdrantClient(url=QDRANT_URL)
    if args.reset:
        try:
            client.delete_collection(COLLECTION)
            log.info("Collection '%s' supprimée.", COLLECTION)
        except Exception:
            pass
        # Recréer via setup_qdrant
        import subprocess
        subprocess.run([sys.executable, "setup_qdrant.py"], check=True)

    # Dry-run
    files = list(iter_documents(root))
    log.info("Fichiers trouvés : %d", len(files))
    if args.dry_run:
        for f in files:
            print(f)
        return

    # Charger le modèle
    log.info("Chargement du modèle sentence-transformers…")
    embedder = Embedder.get_instance()
    log.info("Modèle prêt.")

    total_chunks = 0
    skipped = 0

    for file_path in files:
        rel = file_path.relative_to(root)
        doc_id = hashlib.md5(str(file_path).encode()).hexdigest()

        # Vérifier si déjà indexé (par doc_id prefix dans les IDs)
        # On utilise le nom du fichier comme titre
        title = file_path.stem
        suffix = file_path.suffix.lower()
        doc_type: DocType = (
            DocType.PDF   if suffix == ".pdf"   else
            DocType.DOCX  if suffix in {".docx", ".doc"} else
            DocType.TXT   if suffix == ".txt"   else
            DocType.MD
        )

        text = extract_text(file_path)
        if not text.strip():
            log.warning("Texte vide, ignoré : %s", rel)
            skipped += 1
            continue

        chunks = chunk_text(text)
        log.info("  %s → %d chunks", rel, len(chunks))

        points: list[PointStruct] = []
        texts_batch = chunks  # on va vectoriser en batch

        embeddings = embedder.encode_batch(texts_batch, batch_size=BATCH_SIZE)

        for idx, (chunk_text_item, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_{idx}"
            kw_pairs = _yake.extract_keywords(chunk_text_item)
            keywords = [kw for kw, _ in kw_pairs]
            sp_indices, sp_values = build_sparse(chunk_text_item)

            point = PointStruct(
                id=int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2**63),
                vector={
                    "dense": embedding.tolist(),
                    "sparse": SparseVector(indices=sp_indices, values=sp_values),
                },
                payload={
                    "doc_id":      doc_id,
                    "chunk_id":    chunk_id,
                    "title":       title,
                    "path":        str(rel),
                    "doc_type":    doc_type.value,
                    "content":     chunk_text_item,
                    "keywords":    keywords,
                    "chunk_index": idx,
                },
            )
            points.append(point)

        # Upsert par batch
        for i in range(0, len(points), BATCH_SIZE):
            client.upsert(collection_name=COLLECTION, points=points[i : i + BATCH_SIZE])

        total_chunks += len(points)

    log.info("Indexation terminée — %d chunks insérés, %d fichiers ignorés.", total_chunks, skipped)


if __name__ == "__main__":
    main()
