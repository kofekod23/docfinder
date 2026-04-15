"""
Modèles Pydantic partagés — shared/schema.py.

Ce module est le contrat de données entre le serveur local (server/) et le notebook Colab.
Il définit les structures qui transitent via l'API REST (NDJSON /chunks → /admin/upsert).

Hiérarchie des types :
- DocumentChunk  : représentation d'un chunk prêt à indexer (Colab → Qdrant)
- SearchQuery    : paramètres d'une recherche utilisateur (frontend → /search)
- SearchResult   : résultat enrichi retourné au frontend (/search → UI)

Note : DocType.WORD vs DocType.DOCX — le serveur utilise "docx" en string,
le script local_indexer.py utilise l'enum DocType. Les deux doivent rester alignés.
"""
from enum import Enum
from typing import List

from pydantic import BaseModel


class DocType(str, Enum):
    """Types de documents supportés."""
    PDF = "pdf"
    WORD = "word"
    TXT = "txt"
    MARKDOWN = "md"


class DocumentChunk(BaseModel):
    """
    Chunk d'un document prêt à être indexé dans Qdrant.
    Un document est découpé en plusieurs chunks pour respecter
    la fenêtre de contexte du modèle (512 tokens max).
    """
    doc_id: str        # identifiant unique du document (hash du chemin)
    chunk_id: str      # identifiant unique du chunk : "{doc_id}_{chunk_index}"
    title: str         # nom du fichier sans extension
    path: str          # chemin absolu du fichier source
    doc_type: DocType  # type de document détecté
    content: str       # texte brut du chunk
    keywords: List[str]  # mots-clés extraits par YAKE (top-10)
    chunk_index: int   # position du chunk dans le document (0-indexé)


class SearchQuery(BaseModel):
    """Paramètres d'une requête de recherche."""
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    """Résultat de recherche enrichi avec score RRF et extrait."""
    chunk_id: str
    doc_id: str
    title: str
    path: str
    abs_path: str = ""  # chemin absolu (vide pour les anciens chunks sans ce champ)
    doc_type: str
    score: float       # score RRF fusionné (dense + sparse)
    excerpt: str       # extrait pertinent sélectionné par score de mots-clés (search.py:_best_excerpt)
    keywords: List[str]  # mots-clés du chunk (pour l'affichage)
