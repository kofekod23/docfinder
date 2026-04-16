"""
Moteur de recherche hybride dense + sparse avec reranking RRF.

Principe de fonctionnement :
  1. La requête est encodée en vecteur dense (CPU local, modèle chargé au démarrage).
  2. Les mots-clés de la requête sont extraits par YAKE et convertis en vecteur sparse.
  3. Qdrant local est interrogé deux fois (dense + sparse) avec MAX_CANDIDATES résultats.
  4. Les deux listes sont fusionnées par RRF (Reciprocal Rank Fusion).
  5. Les N meilleurs résultats sont retournés avec leur extrait de contenu.

Aucun appel externe au moment de la recherche.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import yake
from qdrant_client import QdrantClient
from qdrant_client.models import NamedSparseVector, NamedVector, SparseVector

from shared.embedder import QUERY_PREFIX, Embedder
from shared.schema import SearchResult

# Configuration
COLLECTION_NAME = "docfinder"
QDRANT_URL = "http://localhost:6333"
RRF_K = 60           # constante RRF standard (valeur éprouvée dans la littérature)
MAX_CANDIDATES = 50  # candidats récupérés par canal avant fusion

# Extracteur YAKE instancié une seule fois (statistiques locales, pas de réseau)
_yake_extractor = yake.KeywordExtractor(
    lan="fr",
    n=3,         # n-grammes jusqu'à 3 mots
    dedupLim=0.7,
    top=20,
)


def _keyword_to_sparse_index(keyword: str) -> int:
    """
    Hash un mot-clé en index entier pour le vecteur sparse.
    Espace de 2^20 ≈ 1M d'entrées — collisions rares et acceptables.
    """
    return int(hashlib.md5(keyword.lower().encode()).hexdigest(), 16) % (2 ** 20)


def _build_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
    """
    Construit un vecteur sparse à partir des mots-clés YAKE du texte.

    Score YAKE : plus bas = plus important → on inverse pour obtenir l'importance.
    Le vecteur résultant est normalisé entre 0 et 1.

    Returns:
        Tuple (indices, valeurs) du vecteur sparse, ou ([], []) si aucun mot-clé.
    """
    kw_list = _yake_extractor.extract_keywords(text)
    if not kw_list:
        return [], []

    indices: List[int] = []
    values: List[float] = []
    seen: set[int] = set()

    for kw, score in kw_list:
        idx = _keyword_to_sparse_index(kw)
        if idx in seen:
            continue  # évite les doublons dus aux collisions
        seen.add(idx)
        # Plus le score YAKE est bas, plus le mot-clé est important
        importance = 1.0 / (score + 1e-9)
        indices.append(idx)
        values.append(importance)

    # Normalisation min-max
    if values:
        max_val = max(values)
        values = [v / max_val for v in values]

    return indices, values


def _best_excerpt(content: str, keywords: list[str], max_chars: int = 300) -> str:
    """
    Sélectionne les phrases les plus riches en mots-clés pour l'extrait.
    Si aucun mot-clé ne matche, retourne le début du contenu.
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', content) if s.strip()]
    if not sentences:
        return content[:max_chars].strip()

    kw_lower = [k.lower() for k in keywords]

    def score(s: str) -> int:
        sl = s.lower()
        return sum(1 for k in kw_lower if k in sl)

    ranked = sorted(sentences, key=score, reverse=True)
    excerpt = ""
    for s in ranked:
        candidate = (excerpt + " " + s).strip() if excerpt else s
        if len(candidate) > max_chars:
            break
        excerpt = candidate
        if len(excerpt) >= max_chars // 2:
            break

    if not excerpt:
        excerpt = sentences[0]

    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "…"
    return excerpt


def _rrf_fusion(
    dense_hits: List[Tuple[str, float]],
    sparse_hits: List[Tuple[str, float]],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion de deux listes de résultats classés.

    Formule : score(d) = Σ_canal 1 / (k + rang(d, canal))
    Les rangs sont 0-indexés → +1 dans le dénominateur.

    Args:
        dense_hits: Liste (chunk_id, score) triée par score dense décroissant.
        sparse_hits: Liste (chunk_id, score) triée par score sparse décroissant.
        k: Constante de lissage (défaut : 60).

    Returns:
        Liste (chunk_id, score_rrf) triée par score RRF décroissant.
    """
    scores: Dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(dense_hits):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(sparse_hits):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class SearchEngine:
    """
    Moteur de recherche hybride dense + sparse.
    Instancié une seule fois au démarrage du serveur FastAPI.
    """

    def __init__(self) -> None:
        # Récupère le singleton Embedder (chargé au démarrage via lifespan)
        self.embedder = Embedder.get_instance()
        self.client = QdrantClient(url=QDRANT_URL)
        print(f"[SearchEngine] Connecté à Qdrant ({QDRANT_URL}), collection '{COLLECTION_NAME}'.")

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Recherche hybride : dense + sparse avec fusion RRF.

        Args:
            query: Requête en langage naturel.
            limit: Nombre maximum de résultats retournés.

        Returns:
            Liste de SearchResult triés par score RRF décroissant.
        """
        # 1. Encodage dense de la requête (local CPU, ~50ms)
        # Le préfixe "query: " est requis par multilingual-e5 pour la recherche.
        dense_vector = self.embedder.encode(query, prefix=QUERY_PREFIX).tolist()

        # 2. Vecteur sparse de la requête via YAKE
        sparse_indices, sparse_values = _build_sparse_vector(query)

        # 3. Recherche dense dans Qdrant
        dense_hits_raw = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name="dense", vector=dense_vector),
            limit=MAX_CANDIDATES,
            with_payload=True,
        )

        # 4. Recherche sparse (uniquement si YAKE a extrait des mots-clés)
        sparse_hits_raw = []
        if sparse_indices:
            sparse_hits_raw = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                ),
                limit=MAX_CANDIDATES,
                with_payload=True,
            )

        # 5. Conversion en listes (chunk_id, score) pour RRF
        dense_list = [(str(h.id), h.score) for h in dense_hits_raw]
        sparse_list = [(str(h.id), h.score) for h in sparse_hits_raw]

        # 6. Fusion RRF
        fused = _rrf_fusion(dense_list, sparse_list)[:limit]

        # 7. Reconstruction des résultats avec payload Qdrant
        payload_map: Dict[str, dict] = {}
        for hit in dense_hits_raw + sparse_hits_raw:
            payload_map[str(hit.id)] = hit.payload or {}

        results: List[SearchResult] = []
        for chunk_id, rrf_score in fused:
            payload = payload_map.get(chunk_id)
            if not payload:
                continue

            content = payload.get("content", "")
            kw = payload.get("keywords", [])
            excerpt = _best_excerpt(content, kw)

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=payload.get("doc_id", ""),
                    title=payload.get("title", "Sans titre"),
                    path=payload.get("path", ""),
                    abs_path=payload.get("abs_path", ""),
                    doc_type=payload.get("doc_type", ""),
                    score=round(rrf_score, 4),
                    excerpt=excerpt,
                    keywords=payload.get("keywords", []),
                )
            )

        return results
