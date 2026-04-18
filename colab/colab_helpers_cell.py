"""Colab notebook helper functions for DocFinder v2 initialization.

This module collects Colab-specific helpers used during init/run cells.

Environment variables:
  DOCFINDER_ENABLE_QWEN: If "1", load QwenEmbedderWrapper at startup.
  DOCFINDER_EMBEDDER: Embedder selection ("bgem3" | "qwen").
  DOCFINDER_COLLECTION: Qdrant collection to target.
  DOCFINDER_FORCE_REINDEX: Force full re-indexation.
  DOCFINDER_MAX_CHUNKS_PER_DOC: Max chunks per document.
"""

from __future__ import annotations

import os
from typing import Any

from colab.qwen_embedder import QwenEmbedderWrapper


def load_qwen_wrapper_if_enabled(
    query_server_module: Any,
) -> QwenEmbedderWrapper | None:
    """Load and initialize QwenEmbedderWrapper if DOCFINDER_ENABLE_QWEN=1.

    Checks the DOCFINDER_ENABLE_QWEN environment variable. If set to "1",
    creates a QwenEmbedderWrapper, builds the model, registers it with the
    query_server module, and returns it. Otherwise returns None early.

    Args:
        query_server_module: The imported `colab.query_server` module object,
            which exposes `set_qwen_wrapper()` for registration.

    Returns:
        QwenEmbedderWrapper | None: The initialized wrapper if enabled,
            None otherwise.
    """
    if os.environ.get("DOCFINDER_ENABLE_QWEN", "").strip() != "1":
        return None

    wrapper = QwenEmbedderWrapper()
    wrapper._model_or_build()
    query_server_module.set_qwen_wrapper(wrapper)
    print("[helpers] Qwen wrapper loaded.")
    return wrapper
