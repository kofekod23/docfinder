"""Crée la collection Qdrant parallèle `docfinder_v2_qwen` (dense-only).

Usage :
    python -m scripts.setup_qdrant_qwen --variant 0.6B

Pourquoi une collection parallèle :
  - On garde `docfinder_v2` (BGE-M3 hybride) intact pour le A/B.
  - Qwen n'a qu'un canal dense : pas de sparse/colbert à configurer.
  - La dimension dépend de la variante Qwen choisie.
"""

from __future__ import annotations

import argparse
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

DIM_BY_VARIANT: dict[str, int] = {
    "0.6B": 1024,
    "4B": 2560,
    "8B": 4096,
}

DEFAULT_COLLECTION = "docfinder_v2_qwen"
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def create_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    variant: str,
) -> None:
    """Crée la collection si elle n'existe pas déjà (idempotent)."""
    if client.collection_exists(collection_name=collection_name):
        print(f"[setup] collection {collection_name} exists, skip")
        return

    dim = DIM_BY_VARIANT[variant]
    vectors_config = {
        "dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    }
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )
    print(f"[setup] created {collection_name} dim={dim} variant={variant}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="0.6B", choices=list(DIM_BY_VARIANT))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client = QdrantClient(url=args.qdrant_url)
    create_collection(
        client,
        collection_name=args.collection,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
