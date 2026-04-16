# tests/test_setup_qdrant_v2.py
from unittest.mock import MagicMock
from scripts.setup_qdrant_v2 import ensure_collection


def test_ensure_collection_creates_when_missing():
    client = MagicMock()
    client.collection_exists.return_value = False
    ensure_collection(client, "docfinder_v2")
    assert client.create_collection.called
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "docfinder_v2"
    assert "dense" in kwargs["vectors_config"]
    assert "colbert" in kwargs["vectors_config"]
    assert "sparse" in kwargs["sparse_vectors_config"]


def test_ensure_collection_idempotent():
    client = MagicMock()
    client.collection_exists.return_value = True
    ensure_collection(client, "docfinder_v2")
    assert not client.create_collection.called


def test_ensure_payload_indexes():
    client = MagicMock()
    client.collection_exists.return_value = False
    ensure_collection(client, "docfinder_v2")
    keys = [c.kwargs["field_name"] for c in client.create_payload_index.call_args_list]
    assert {"doc_id", "doc_type", "mtime", "path"}.issubset(keys)
