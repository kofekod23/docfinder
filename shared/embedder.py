"""
Wrapper autour du modèle sentence-transformers.

Le modèle est chargé une seule fois via le singleton Embedder.get_instance().
Au démarrage du serveur FastAPI, SearchEngine() appelle get_instance() qui
déclenche le chargement. Toutes les requêtes de recherche réutilisent ensuite
le même objet en mémoire — aucun appel externe, CPU uniquement.

Ce module est aussi importable depuis le notebook Colab (GPU).
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Désactive le parallélisme des tokenizers HuggingFace pour éviter les forks
# orphelins sous macOS (multiprocessing.spawn_main zombies qui saturent le CPU).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Modèle multilingue haute qualité — 1024 dimensions, ~560 MB
# Requiert les préfixes : "query: " pour les recherches, "passage: " pour l'indexation.
MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder:
    """
    Encapsule le modèle d'embedding.
    Utiliser Embedder.get_instance() plutôt que le constructeur directement.
    """

    _instance: "Embedder | None" = None

    def __init__(self) -> None:
        """Charge le modèle en mémoire (~10s, ~1 GB RAM)."""
        print(f"[Embedder] Chargement du modèle '{MODEL_NAME}' (CPU)…")
        # Forcer CPU : on encode uniquement des requêtes unitaires en local.
        # Évite de mobiliser le GPU Metal (MPS) pour rien sur Apple Silicon.
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        print(f"[Embedder] Modèle prêt — {EMBEDDING_DIM} dims.")

    @classmethod
    def get_instance(cls) -> "Embedder":
        """Retourne l'instance singleton (chargement paresseux)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, text: str, prefix: str = PASSAGE_PREFIX) -> np.ndarray:
        """
        Encode un texte en vecteur dense L2-normalisé.

        Args:
            text:   Texte à encoder (requête ou contenu de chunk).
            prefix: Préfixe à antéposer — QUERY_PREFIX pour les recherches,
                    PASSAGE_PREFIX pour l'indexation (défaut).

        Returns:
            Vecteur numpy de taille EMBEDDING_DIM.
        """
        return self.model.encode(prefix + text, normalize_embeddings=True)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        prefix: str = PASSAGE_PREFIX,
    ) -> np.ndarray:
        """
        Encode un batch de textes (plus efficace sur GPU pour l'indexation).

        Args:
            texts:      Liste de textes à encoder.
            batch_size: Taille des mini-batches internes.
            prefix:     Préfixe à antéposer à chaque texte.

        Returns:
            Tableau numpy (N, EMBEDDING_DIM) normalisé.
        """
        prefixed = [prefix + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
