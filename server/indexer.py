"""
Indexation en arrière-plan — server/indexer.py.

Moteur d'indexation LOCAL du serveur FastAPI. Invoqué via POST /admin/index.
Tourne dans un thread daemon (un seul job à la fois) pour ne pas bloquer l'event loop asyncio.

Flux de données :
    fichiers disque → extraction texte → chunking → YAKE sparse vectors
    → embeddings CPU (Embedder singleton) → upsert Qdrant local

Responsabilités :
- Scanner un répertoire récursivement (PDF, DOCX, TXT, MD)
- Extraire le texte brut (pymupdf pour PDF, python-docx pour DOCX)
- Découper en chunks de 1500 chars avec overlap 200
- Générer les vecteurs dense 768-dim (paraphrase-multilingual-mpnet-base-v2) et sparse (YAKE+MD5)
- Insérer dans Qdrant par batches de 32 points (upsert idempotent)
- Exposer l'état du job (progression, logs, statut) via current_job()

Différence avec les modules voisins :
- server/chunks.py  : génère un flux NDJSON pour Colab (GPU) — n'insère pas dans Qdrant
- local_indexer.py  : script CLI autonome (même logique, sans serveur web)
- shared/embedder.py: singleton sentence-transformers, forcé sur CPU (device="cpu")
"""
from __future__ import annotations

import hashlib
import json
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

# Fichier de persistance : si présent au démarrage du serveur → reprendre le job
_RESUME_FILE = Path(__file__).parent.parent / "storage" / "job_resume.json"


def _save_resume(path: str) -> None:
    """Persiste le chemin du job en cours pour reprise automatique au redémarrage."""
    try:
        _RESUME_FILE.parent.mkdir(parents=True, exist_ok=True)
        _RESUME_FILE.write_text(json.dumps({"path": path}))
    except OSError:
        log.warning("Impossible d'écrire %s", _RESUME_FILE)


def _clear_resume() -> None:
    """Supprime le fichier de reprise (job terminé normalement ou annulé)."""
    try:
        _RESUME_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def resume_if_needed() -> None:
    """À appeler au démarrage du serveur : relance le job si crash précédent."""
    if not _RESUME_FILE.exists():
        return
    try:
        data = json.loads(_RESUME_FILE.read_text())
        path = data.get("path", "")
        if path:
            log.info("[resume] Job interrompu détecté — reprise : %s", path)
            start_indexation(path=path, reset=False)
    except Exception:
        log.warning("[resume] Fichier de reprise illisible, ignoré.", exc_info=True)
        _clear_resume()


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


def _upsert_with_retry(
    client: QdrantClient,
    points: list[PointStruct],
    max_retries: int = 3,
) -> None:
    """Upsert un batch dans Qdrant avec retry exponentiel (1s → 2s → 4s).

    Qdrant peut échouer ponctuellement sur un batch (overload, GC pause).
    3 tentatives couvrent les transitoires sans masquer les pannes durables.
    """
    import time as _time

    delay = 1.0
    for attempt in range(1, max_retries + 2):
        try:
            client.upsert(collection_name=COLLECTION, points=points, wait=False)
            return
        except Exception as e:
            if attempt > max_retries:
                raise
            log.warning(
                "upsert Qdrant échoué (tentative %d/%d) : %s — retry dans %.0fs",
                attempt, max_retries, e, delay,
            )
            _time.sleep(delay)
            delay *= 2


def upsert_points(points_data: list[dict]) -> int:
    """Insère des points (envoyés par Colab) dans Qdrant local.

    Met aussi à jour le job courant (done, chunks, current_file) pour que
    l'UI admin reflète la progression Colab en temps réel.
    """
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
        _upsert_with_retry(client, structs[i : i + BATCH_SIZE])

    # Mise à jour du job pour l'UI
    # Un batch Colab = tous les chunks d'UN fichier (chunk_index 0 = début de fichier)
    if points_data:
        new_files = sum(
            1 for p in points_data if p.get("payload", {}).get("chunk_index", 1) == 0
        )
        last_path = points_data[-1].get("payload", {}).get("path", "")
        with _lock:
            _job.chunks += len(structs)
            _job.done += new_files
            if last_path:
                _job.current_file = last_path

    return len(structs)


def colab_file_skipped(count: int) -> None:
    """Appelé quand Colab signale des fichiers déjà indexés (skip).

    Met à jour _job.done et _job.skipped pour que l'UI reflète la vraie progression.
    """
    with _lock:
        if _job.status == JobStatus.RUNNING:
            _job.done += count
            _job.skipped += count


def colab_job_start(path: str, total: int) -> None:
    """Appelé par /chunks quand Colab démarre une indexation.

    Si aucun job local n'est en cours, bascule _job en RUNNING pour que l'UI
    reflète la progression Colab. N'interfère pas avec un job local actif.
    """
    with _lock:
        if _job.status == JobStatus.RUNNING:
            # Job local actif : juste mettre à jour le total si même chemin
            if _job.path == path:
                _job.total = total
            return
        # Pas de job local — on réinitialise les champs en place (pas de réassignation)
        _job.status = JobStatus.RUNNING
        _job.path = path
        _job.total = total
        _job.done = 0
        _job.chunks = 0
        _job.current_file = ""
        _job.error = ""
        _job.log_lines = [f"Indexation Colab démarrée : {path}"]
    _save_resume(path)


def start_indexation(path: str, reset: bool = False) -> dict:
    """Lance l'indexation dans un thread daemon. Retourne une erreur si déjà en cours."""
    global _job
    with _lock:
        if _job.status == JobStatus.RUNNING:
            return {"error": "Une indexation est déjà en cours."}
        _job = IndexJob(status=JobStatus.RUNNING, path=path)
    _cancel_event.clear()
    _save_resume(path)

    t = threading.Thread(target=_run, args=(path, reset), daemon=True)
    t.start()
    return {"started": True, "path": path}


def cancel_indexation() -> dict:
    """Annule le job en cours — immédiatement pour les jobs Colab, via event pour les jobs locaux."""
    with _lock:
        if _job.status != JobStatus.RUNNING:
            return {"error": "Aucune indexation en cours."}
        # Force la transition immédiate : couvre les jobs Colab (pas de thread local)
        # et les jobs locaux (le thread verra aussi _cancel_event et confirme l'erreur).
        _job.status = JobStatus.ERROR
        _job.error = "Annulé par l'utilisateur."
        _job.current_file = ""
    _cancel_event.set()
    _clear_resume()
    return {"cancelled": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Logue msg dans le logger standard ET dans le buffer du job courant (thread-safe)."""
    log.info(msg)
    with _lock:
        _job.log_lines.append(msg)


def _kw_index(kw: str) -> int:
    """Hash MD5 d'un mot-clé → index entier dans l'espace sparse [0, 2^20).

    Même logique que chunks.py et local_indexer.py — tous les modules doivent
    rester synchronisés pour que les vecteurs sparse soient comparables.
    Collisions possibles (~1/1M) : acceptables pour ce cas d'usage (D3).
    """
    return int(hashlib.md5(kw.lower().encode()).hexdigest(), 16) % (2 ** 20)


def _build_sparse(text: str) -> tuple[list[int], list[float]]:
    """Construit un vecteur sparse à partir du texte via YAKE.

    Étapes :
    1. YAKE extrait les n-grammes (n≤3) avec leur score de saillance.
    2. Le score YAKE est *inversé* (1/score) pour que les meilleurs mots-clés aient la valeur la plus haute.
    3. Les valeurs sont normalisées dans [0, 1] par rapport au maximum.
    4. Chaque mot-clé est hashé via _kw_index() → indice Qdrant SparseVector.

    Retourne (indices, values) compatibles avec qdrant_client.models.SparseVector.
    """
    pairs = _yake_extractor.extract_keywords(text)
    if not pairs:
        return [], []
    raw = {_kw_index(kw): 1.0 / (s + 1e-9) for kw, s in pairs}
    max_v = max(raw.values())
    return list(raw.keys()), [v / max_v for v in raw.values()]


def _extract_text(path: Path) -> str:
    """Extrait le texte brut d'un fichier selon son extension.

    Parseurs utilisés :
    - .pdf   → pymupdf (fitz) — page.get_text() sur chaque page
    - .docx  → python-docx   — paragraphes joints par \\n
    - .txt/.md → lecture directe (errors="ignore" pour les encodages cassés)

    Retourne "" en cas d'échec (fichier chiffré, corrompu, encodage inconnu).
    Le fallback sur le chemin fichier est géré dans _run(), pas ici.
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
        _log(f"  ⚠ extraction échouée : {e}")
    return ""


def _split_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases sur les ponctuations finales."""
    parts = re.split(r"(?<=[.!?»])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _chunk(text: str) -> list[str]:
    """Découpage hybride : paragraphes → phrases → overlap sémantique.

    Synchronisé avec server/chunks.py — toute modification doit être répercutée.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    raw: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > CHUNK_SIZE:
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

    result: list[str] = [raw[0]]
    for i in range(1, len(raw)):
        tail = raw[i - 1][-CHUNK_OVERLAP:]
        m = re.search(r"(?<=[\.\!\?»])\s+\S", tail)
        if m:
            tail = tail[m.start():].strip()
        elif len(tail) > CHUNK_OVERLAP // 2:
            sp = tail.find(" ")
            tail = tail[sp + 1:].strip() if sp != -1 else ""
        else:
            tail = ""
        result.append((tail + " " + raw[i]).strip() if tail else raw[i])

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


def _iter_files(root: Path):
    """Générateur récursif — yield les fichiers indexables sous root.

    Exclusions :
    - .icloud  : placeholders iCloud non téléchargés (contenu absent)
    - ~$*      : fichiers temporaires Office (Word, Excel ouvert)
    - .*       : fichiers/dossiers cachés macOS (.DS_Store, .git, etc.)
    """
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
                _clear_resume()
                return

            rel = str(file_path.relative_to(root))
            with _lock:
                _job.current_file = rel

            text = _extract_text(file_path)
            hint = _name_hint(file_path)
            if not text.strip():
                # Texte vide (PDF scanné, fichier chiffré…) → chemin comme contenu
                text = hint
                _log(f"  {rel} → texte vide, chemin utilisé comme contenu")
            else:
                # Préfixe le nom pour que YAKE et l'embedding voient toujours les
                # mots-clés du fichier, même quand le texte principal est générique.
                text = hint + "\n" + text

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
                _upsert_with_retry(client, points[i: i + BATCH_SIZE])

            with _lock:
                _job.chunks += len(points)
                _job.done += 1

        _log(f"Terminé — {_job.chunks} chunks insérés, {_job.skipped} fichiers ignorés.")
        with _lock:
            # Ne pas écraser un ERROR forcé par cancel_indexation()
            if _job.status == JobStatus.RUNNING:
                _job.status = JobStatus.DONE
                _job.current_file = ""
        _clear_resume()

    except Exception as e:
        log.exception("Erreur indexation")
        with _lock:
            _job.status = JobStatus.ERROR
            _job.error = str(e)
        # Pas de _clear_resume() ici : le fichier reste pour permettre la reprise au redémarrage
