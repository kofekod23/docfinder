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
import re
import unicodedata
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


# Mots trop génériques pour discriminer un document : ils bruiteraient
# le boost filename/titre (ex. "documents médicaux" → matche tout fichier
# contenant "documents" dans le path, même sans lien avec le sujet médical).
_STOPWORDS_BOOST: frozenset[str] = frozenset({
    # FR
    "document", "documents", "fichier", "fichiers", "dossier", "dossiers",
    "liste", "page", "pages", "annexe", "annexes", "copie", "copies",
    "version", "extrait", "note", "notes",
    # EN
    "document", "file", "files", "folder", "folders", "list", "page",
    "pages", "annex", "copy", "note", "notes", "and", "the", "for", "with",
})


def _extract_boost_tokens(query: str) -> set[str]:
    """Extract filename-boost tokens from a query.

    Admission rules:
      - normalized token of length >= 4 and not in stopwords, OR
      - normalized token of length 2-3 if it was fully uppercase in the
        original query (acronym heuristic: PV, SEO, CV, RH).

    Returns a set of accent-stripped lowercase tokens.
    """
    uppercase_tokens = {
        _normalize_text(w) for w in re.findall(r"\b[A-Z]{2,}\b", query)
    }
    tokens: set[str] = set()
    for t in _normalize_text(query).split():
        if t in _STOPWORDS_BOOST:
            continue
        if len(t) >= 4:
            tokens.add(t)
        elif 2 <= len(t) <= 3 and t in uppercase_tokens:
            tokens.add(t)
    return tokens


def _normalize_text(s: str) -> str:
    """Lower + strip accents (NFD) + replace non-alphanumeric runs with spaces.

    Used for tolerant filename/title matching: "médicaments" → "medicaments",
    "ordonnance-de-medicaments_2026.pdf" → "ordonnance de medicaments 2026 pdf".
    """
    nfd = unicodedata.normalize("NFD", s.lower())
    no_accents = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", no_accents).strip()


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
        self.qdrant = self.client  # Alias for compatibility with v2 path
        self._embedder_v2 = None  # Lazy-loaded BGE-M3 embedder for v2 search
        print(f"[SearchEngine] Connecté à Qdrant ({QDRANT_URL}), collection '{COLLECTION_NAME}'.")

    @property
    def embedder_v2(self):
        """Lazy-load BGE-M3 encoder for v2 search.

        Mac stateless (spec v2) : si `COLAB_ENCODE_URL` est défini on passe par
        `RemoteEncoder` (HTTP → serveur Colab). Sinon on tente le wrapper local
        (ne fonctionnera pas sur Intel Mac mais reste utile en test unitaire).
        """
        if self._embedder_v2 is None:
            import os
            remote_url = os.environ.get("COLAB_ENCODE_URL", "").strip()
            try:
                if remote_url:
                    from server.encode_client import RemoteEncoder
                    self._embedder_v2 = RemoteEncoder()
                    print(f"[SearchEngine] embedder_v2 = RemoteEncoder({remote_url})")
                else:
                    from colab.embedder_v2 import BGEM3Wrapper
                    self._embedder_v2 = BGEM3Wrapper()
                    print("[SearchEngine] embedder_v2 = BGEM3Wrapper (local)")
            except Exception as exc:
                print(f"[SearchEngine] embedder_v2 load failed: {exc}")
                self._embedder_v2 = None
        return self._embedder_v2

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
              prefetch_limit: int = 300) -> list[SearchResult]:
    """Fusion RRF au niveau **document** (pas chunk).

    Chaque canal (dense/sparse/colbert) est interrogé séparément; on
    dédoublonne par doc_id en gardant le meilleur chunk du doc, PUIS on
    fusionne par RRF. Sans ce garde-fou, un gros document multi-chunks
    domine: il a N chances d'apparaître en tête dans chaque canal face
    à un document 1-chunk pertinent qui ne peut apparaître qu'une fois.

    Qdrant `FusionQuery(RRF)` natif fusionne au niveau chunk — un doc
    de 20 chunks accumule 20 entrées alors qu'un doc de 1 chunk en a 1,
    d'où disparition des petits documents pertinents du top-N.
    """
    enc = embedder.encode([query])
    dense_q = enc.dense[0]
    sparse_q = qm.SparseVector(indices=enc.sparse[0][0], values=enc.sparse[0][1])
    colbert_q = enc.colbert[0]

    def _rank_docs(q, using: str) -> list[tuple[str, int, object]]:
        """Retourne [(doc_id, rank_0based, point)] dédoublonné: 1 chunk/doc."""
        resp = qdrant.query_points(
            collection_name=collection, query=q, using=using,
            limit=prefetch_limit, with_payload=True,
        )
        seen: dict[str, tuple[int, object]] = {}
        for rank, pt in enumerate(resp.points):
            pl = pt.payload or {}
            doc_id = pl.get("doc_id") or str(pt.id)
            if doc_id not in seen:
                seen[doc_id] = (rank, pt)
        return [(d, r, p) for d, (r, p) in seen.items()]

    channels = (
        _rank_docs(dense_q, "dense"),
        _rank_docs(sparse_q, "sparse"),
        _rank_docs(colbert_q, "colbert"),
    )

    rrf: dict[str, float] = {}
    best_hit: dict[str, object] = {}
    best_rank: dict[str, int] = {}
    for channel in channels:
        for doc_id, rank, pt in channel:
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
                best_hit[doc_id] = pt

    # Filename/title boost — signal très fiable en recherche documentaire.
    # Un token de la requête qui apparaît dans le path/title récolte 1/RRF_K,
    # soit l'équivalent d'un rang #1 dans un canal virtuel "filename".
    #
    # Règles d'admission des tokens :
    #   - len >= 4 et pas dans STOPWORDS (filtre "documents", "liste", etc.)
    #   - OU len 2-3 si le token était UPPERCASE dans la requête d'origine
    #     (heuristique pour acronymes : PV, SEO, CV, RH)
    #
    # Pondération par rareté : si un token apparaît dans > RARITY_THRESHOLD
    # du pool (best_hit), son poids est divisé par 2 — il discrimine mal.
    RARITY_THRESHOLD = 0.30
    q_tokens = _extract_boost_tokens(query)
    if q_tokens and best_hit:
        pool_size = len(best_hit)
        pool_haystacks = {
            doc_id: _normalize_text(
                f"{(getattr(pt, 'payload', None) or {}).get('path', '')} "
                f"{(getattr(pt, 'payload', None) or {}).get('title', '')}"
            )
            for doc_id, pt in best_hit.items()
        }
        token_weights: dict[str, float] = {}
        for t in q_tokens:
            df = sum(1 for h in pool_haystacks.values() if t in h)
            base = 1.0 / RRF_K
            token_weights[t] = base * 0.5 if df / pool_size > RARITY_THRESHOLD else base
        for doc_id, haystack in pool_haystacks.items():
            boost = sum(w for t, w in token_weights.items() if t in haystack)
            if boost:
                rrf[doc_id] += boost

    top = sorted(rrf.items(), key=lambda x: -x[1])[:limit]
    out: list[SearchResult] = []
    for doc_id, score in top:
        pt = best_hit[doc_id]
        pl = getattr(pt, "payload", None) or {}
        out.append(SearchResult(
            chunk_id=str(pt.id),
            doc_id=doc_id,
            title=pl.get("title", ""),
            path=pl.get("path", ""),
            abs_path=pl.get("abs_path", ""),
            doc_type=pl.get("doc_type", ""),
            score=round(score, 4),
            excerpt=pl.get("content", ""),
            keywords=pl.get("keywords_doc", []),
        ))
    return out
