"""
Initialise la collection Qdrant pour DocFinder.

À lancer UNE SEULE FOIS avant la première indexation :
    python setup_qdrant.py

Crée la collection "docfinder" avec :
  - Vecteurs denses : 768 dims, distance cosine (paraphrase-multilingual-mpnet-base-v2)
  - Vecteurs sparse : indices + valeurs YAKE (BM25 maison)
"""
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

# Configuration Qdrant local (binaire natif)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "docfinder"
DENSE_DIM = 768  # paraphrase-multilingual-mpnet-base-v2


def setup_collection(force: bool = False) -> None:
    """
    Crée ou recrée la collection Qdrant.

    Args:
        force: Si True, supprime la collection existante avant de la recréer.
    """
    client = QdrantClient(url=QDRANT_URL)

    # Vérification de la connexion
    try:
        client.get_collections()
    except Exception as exc:
        print(f"[ERREUR] Impossible de joindre Qdrant sur {QDRANT_URL}")
        print(f"         Assurez-vous que le binaire Qdrant est démarré.")
        print(f"         Détail : {exc}")
        sys.exit(1)

    # Gestion de la collection existante
    collections = client.get_collections().collections
    existe = any(c.name == COLLECTION_NAME for c in collections)

    if existe:
        if force:
            print(f"[INFO] Collection '{COLLECTION_NAME}' existante — suppression…")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"[INFO] Collection '{COLLECTION_NAME}' déjà présente.")
            print("       Utilisez --force pour la recréer.")
            return

    # Création de la collection avec vecteurs denses + sparse
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            )
        },
    )

    print(f"[OK] Collection '{COLLECTION_NAME}' créée avec succès.")
    print(f"     - Vecteurs denses  : {DENSE_DIM} dims, distance cosine")
    print(f"     - Vecteurs sparse  : YAKE/BM25, in-memory index")
    print(f"     - URL Qdrant       : {QDRANT_URL}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    setup_collection(force=force)
