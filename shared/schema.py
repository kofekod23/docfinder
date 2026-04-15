"""
Modèles Pydantic partagés entre le serveur local et le notebook Colab.
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
    excerpt: str       # 300 premiers caractères du chunk pour l'affichage
    keywords: List[str]  # mots-clés du chunk (pour l'affichage)
