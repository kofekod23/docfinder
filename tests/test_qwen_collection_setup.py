"""Tests du script de création de collection Qdrant pour Qwen."""

from __future__ import annotations

from unittest.mock import MagicMock

from scripts.setup_qdrant_qwen import create_collection, DIM_BY_VARIANT


def test_dim_by_variant_covers_all_expected() -> None:
    assert DIM_BY_VARIANT["0.6B"] == 1024
    assert DIM_BY_VARIANT["4B"] == 2560
    assert DIM_BY_VARIANT["8B"] == 4096


def test_create_collection_creates_dense_only() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    create_collection(
        client,
        collection_name="docfinder_v2_qwen",
        variant="0.6B",
    )

    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "docfinder_v2_qwen"
    assert "dense" in kwargs["vectors_config"]
    assert kwargs["vectors_config"]["dense"].size == 1024
    assert kwargs.get("sparse_vectors_config") in (None, {})


def test_create_collection_skip_if_exists() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True

    create_collection(
        client,
        collection_name="docfinder_v2_qwen",
        variant="0.6B",
    )

    client.create_collection.assert_not_called()
