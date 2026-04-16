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
from qdrant_client.http import models as qm
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


def _best_excerpt(content: str, query_keywords: list[str], max_chars: int = 300) -> str:
    """
    Sélectionne les phrases les plus riches en mots-clés de la *requête*.
    Si aucun mot-clé ne matche, retourne le début du contenu.

    Args:
        content:        Texte complet du chunk.
        query_keywords: Mots-clés extraits de la requête utilisateur (pas du document).
        max_chars:      Longueur maximale de l'extrait retourné.
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', content) if s.strip()]
    if not sentences:
        return content[:max_chars].strip()

    kw_lower = [k.lower() for k in query_keywords]

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

        Un seul résultat par document (doc_id) est retourné — le chunk dont le
        score RRF est le plus élevé. Cela évite qu'un document avec beaucoup de
        chunks monopolise le top-N.

        Args:
            query: Requête en langage naturel.
            limit: Nombre maximum de résultats retournés (un par document).

        Returns:
            Liste de SearchResult triés par score RRF décroissant.
        """
        # 1. Encodage dense de la requête (local CPU, ~50ms)
        # Le préfixe "query: " est requis par multilingual-e5 pour la recherche.
        dense_vector = self.embedder.encode(query, prefix=QUERY_PREFIX).tolist()

        # 2. Vecteur sparse de la requête via YAKE
        sparse_indices, sparse_values = _build_sparse_vector(query)
        # Extraire les mots-clés textuels pour l'extrait (avant normalisation)
        query_kw_pairs = _yake_extractor.extract_keywords(query)
        query_keywords = [kw for kw, _ in query_kw_pairs]

        # 3. Recherche dense dans Qdrant
        # On récupère plus de candidats pour compenser la déduplication par doc.
        candidates = MAX_CANDIDATES * 3
        dense_hits_raw = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name="dense", vector=dense_vector),
            limit=candidates,
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
                limit=candidates,
                with_payload=True,
            )

        # 5. Conversion en listes (chunk_id, score) pour RRF
        dense_list = [(str(h.id), h.score) for h in dense_hits_raw]
        sparse_list = [(str(h.id), h.score) for h in sparse_hits_raw]

        # 6. Fusion RRF (sur tous les candidats, avant déduplication)
        fused = _rrf_fusion(dense_list, sparse_list)

        # 7. Reconstruction des résultats avec payload Qdrant
        payload_map: Dict[str, dict] = {}
        for hit in dense_hits_raw + sparse_hits_raw:
            payload_map[str(hit.id)] = hit.payload or {}

        # 8. Déduplication par document : garde le meilleur chunk par doc_id.
        # RRF est déjà trié par score décroissant → le premier chunk rencontré
        # pour chaque doc_id est toujours le plus pertinent.
        seen_docs: set[str] = set()
        results: List[SearchResult] = []

        for chunk_id, rrf_score in fused:
            if len(results) >= limit:
                break

            payload = payload_map.get(chunk_id)
            if not payload:
                continue

            doc_id = payload.get("doc_id", chunk_id)
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            content = payload.get("content", "")
            # Extrait basé sur les mots-clés de la requête (pas du document)
            # pour maximiser la pertinence perçue.
            excerpt = _best_excerpt(content, query_keywords or payload.get("keywords", []))

            # Mots-clés affichés = mots-clés de la requête présents dans le contenu.
            # Si aucun ne matche, fallback sur les 3 premiers mots-clés de la requête.
            content_lower = content.lower()
            relevant_kw = [kw for kw in query_keywords if kw.lower() in content_lower]
            if not relevant_kw:
                relevant_kw = query_keywords[:3]

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=payload.get("title", "Sans titre"),
                    path=payload.get("path", ""),
                    abs_path=payload.get("abs_path", ""),
                    doc_type=payload.get("doc_type", ""),
                    score=round(rrf_score, 4),
                    excerpt=excerpt,
                    keywords=relevant_kw,
                )
            )

        return results


def search_v2(qdrant, embedder, query: str,
              collection: str = "docfinder_v2",
              limit: int = 10,
              prefetch_limit: int = 50) -> list[SearchResult]:
    """Dense + sparse with RRF, then ColBERT MaxSim rerank."""
    enc = embedder.encode([query])
    dense_q = enc.dense[0]
    sparse_q = qm.SparseVector(indices=enc.sparse[0][0], values=enc.sparse[0][1])
    colbert_q = enc.colbert[0]

    prefetch = [
        qm.Prefetch(query=dense_q, using="dense", limit=prefetch_limit),
        qm.Prefetch(query=sparse_q, using="sparse", limit=prefetch_limit),
    ]

    resp = qdrant.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=colbert_q,
        using="colbert",
        limit=limit,
        with_payload=True,
    )

    out: list[SearchResult] = []
    for pt in resp.points:
        pl = pt.payload or {}
        out.append(SearchResult(
            chunk_id=str(pt.id),
            doc_id=pl.get("doc_id", ""),
            title=pl.get("title", ""),
            path=pl.get("path", ""),
            abs_path=pl.get("abs_path", ""),
            doc_type=pl.get("doc_type", ""),
            score=float(pt.score),
            excerpt=pl.get("content", ""),
            keywords=pl.get("keywords_doc", []),
        ))
    return out
