"""Create docfinder_v2 collection + payload indexes (spec §4)."""
from __future__ import annotations

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def ensure_collection(client: QdrantClient, name: str = "docfinder_v2") -> None:
    if client.collection_exists(collection_name=name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE),
            "colbert": qm.VectorParams(
                size=1024,
                distance=qm.Distance.COSINE,
                multivector_config=qm.MultiVectorConfig(
                    comparator=qm.MultiVectorComparator.MAX_SIM,
                ),
                on_disk=True,
            ),
        },
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
        },
    )
    for field, schema in [
        ("doc_id", qm.PayloadSchemaType.KEYWORD),
        ("doc_type", qm.PayloadSchemaType.KEYWORD),
        ("mtime", qm.PayloadSchemaType.INTEGER),
        ("path", qm.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema,
        )


if __name__ == "__main__":
    client = QdrantClient(
        host=os.environ.get("QDRANT_HOST", "127.0.0.1"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
    )
    ensure_collection(client)
    print("docfinder_v2 ready")
