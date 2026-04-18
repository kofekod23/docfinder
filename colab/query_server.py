"""FastAPI endpoint `/encode` — query encoding server exposed from Colab GPU.

Pourquoi : le Mac Intel x86_64 ne peut pas exécuter BGE-M3 (CVE-2025-32434 force
torch >= 2.6, absent de x86_64). On déporte donc l'encodage des requêtes vers
Colab (T4, torch 2.6+, fp16). Le Mac devient un simple serveur de fichiers qui
forward la requête à ce `/encode` via un tunnel Cloudflare nommé.

Contrat (miroir de `colab.embedder_v2.BGEM3Wrapper.encode`) :
  POST /encode
  Headers: X-Auth-Token: <token>
  Body:    {"queries": ["...", "..."]}
  200:     {"dense": [[...]], "sparse": [[[idx], [val]]], "colbert": [[[...]]]}
  401:     token invalide
  503:     modèle non chargé

Lancement (dans le notebook Colab) :
  import os, uvicorn
  os.environ["COLAB_QUERY_TOKEN"] = "..."
  from colab.query_server import app
  uvicorn.run(app, host="0.0.0.0", port=8001)
"""

from __future__ import annotations

import logging
import os
from typing import List

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from colab.embedder_v2 import BGEM3Wrapper, EncodeResult
from colab.qwen_embedder import QwenEmbedderWrapper
from colab.reranker import BGERerankerWrapper

logger = logging.getLogger("docfinder.query_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


class EncodeRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1, max_length=32)
    batch_size: int = Field(default=32, ge=1, le=128)
    max_length: int = Field(default=512, ge=16, le=8192)


class EncodeResponse(BaseModel):
    dense: List[List[float]]
    sparse: List[List[List[float]]]
    colbert: List[List[List[float]]]


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: List[str] = Field(..., min_length=1, max_length=200)


class RerankResponse(BaseModel):
    scores: List[float]


app = FastAPI(title="DocFinder Query Encoder", version="1.0.0")
_wrapper: BGEM3Wrapper | None = None
_reranker: BGERerankerWrapper | None = None
_qwen_wrapper: QwenEmbedderWrapper | None = None


def _expected_token() -> str:
    token = os.environ.get("COLAB_QUERY_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "COLAB_QUERY_TOKEN must be set in the environment before starting the server."
        )
    return token


def _check_auth(received: str | None) -> None:
    expected = _expected_token()
    if not received or received.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token"
        )


def set_wrapper(wrapper: BGEM3Wrapper) -> None:
    """Inject a pre-built wrapper to avoid loading BGE-M3 twice into GPU memory.

    Utilisé par `colab_helpers_cell.py` pour partager l'instance avec l'indexer.
    Doit être appelé AVANT `uvicorn.run(app, ...)`.
    """
    global _wrapper
    _wrapper = wrapper
    logger.info("BGE-M3 wrapper injected (shared with indexer).")


def set_reranker(reranker: BGERerankerWrapper) -> None:
    """Inject a pre-built reranker to avoid loading bge-reranker-v2-m3 twice.

    Miroir de `set_wrapper` pour le cross-encoder. Doit être appelé AVANT
    `uvicorn.run(app, ...)` si l'indexer partage déjà l'instance GPU.
    """
    global _reranker
    _reranker = reranker
    logger.info("BGE-reranker wrapper injected.")


def set_qwen_wrapper(wrapper: QwenEmbedderWrapper) -> None:
    """Inject a pre-built Qwen wrapper (avoids double-load in Colab).

    Miroir de `set_wrapper` pour Qwen3-Embedding. Doit être appelé AVANT
    `uvicorn.run(app, ...)`.
    """
    global _qwen_wrapper
    _qwen_wrapper = wrapper
    logger.info("Qwen3-Embedding wrapper injected.")


@app.on_event("startup")
def _load_model() -> None:
    global _wrapper, _reranker
    if _wrapper is None:
        logger.info("Loading BGE-M3 model on GPU (fp16)…")
        _wrapper = BGEM3Wrapper()
        _wrapper._model_or_build()
        logger.info("BGE-M3 model ready.")
    else:
        logger.info("BGE-M3 wrapper already present, skipping reload.")
    if _reranker is None and os.environ.get("LOAD_RERANKER", "1") == "1":
        logger.info("Loading BGE-reranker-v2-m3 on GPU (fp16)…")
        _reranker = BGERerankerWrapper()
        _reranker._load()
        logger.info("BGE-reranker ready.")


@app.get("/healthz")
def healthz() -> dict:
    ready = _wrapper is not None and _wrapper._model is not None
    return {"status": "ok" if ready else "loading", "model": "bge-m3"}


@app.post("/encode", response_model=EncodeResponse)
def encode(
    req: EncodeRequest, x_auth_token: str | None = Header(default=None)
) -> EncodeResponse:
    _check_auth(x_auth_token)
    if _wrapper is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model not loaded"
        )

    result: EncodeResult = _wrapper.encode(
        req.queries,
        batch_size=req.batch_size,
        max_length=req.max_length,
    )
    sparse_payload = [
        [list(map(float, idx)), list(map(float, val))] for idx, val in result.sparse
    ]
    return EncodeResponse(
        dense=result.dense, sparse=sparse_payload, colbert=result.colbert
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(
    req: RerankRequest,
    x_auth_token: str | None = Header(default=None),
) -> RerankResponse:
    _check_auth(x_auth_token)
    if _reranker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="reranker not loaded",
        )
    pairs = [(req.query, doc) for doc in req.documents]
    scores = _reranker.rerank(pairs)
    return RerankResponse(scores=scores)


# --- Qwen3-Embedding support (dense-only) --------------------------------


@app.post("/encode_qwen", response_model=EncodeResponse)
def encode_qwen(
    req: EncodeRequest,
    x_auth_token: str | None = Header(default=None),
) -> EncodeResponse:
    _check_auth(x_auth_token)
    if _qwen_wrapper is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="qwen model not loaded",
        )
    result: EncodeResult = _qwen_wrapper.encode(
        req.queries,
        batch_size=req.batch_size,
        max_length=req.max_length,
    )
    # Qwen retourne dense uniquement — sparse/colbert vides
    return EncodeResponse(dense=result.dense, sparse=[], colbert=[])
