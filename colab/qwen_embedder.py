"""Wrapper Qwen3-Embedding — dense-only, compatible `EncodeResult` (BGE-M3).

Pourquoi : Qwen3-Embedding n'expose pas de sparse ni de colbert ; on remplit
les champs correspondants avec des listes vides pour garder la même signature
que `BGEM3Wrapper.encode` et limiter l'impact sur le reste du pipeline.
"""
from __future__ import annotations

from typing import List

from colab.embedder_v2 import EncodeResult

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


class QwenEmbedderWrapper:
    """Encodeur Qwen3-Embedding dense-only.

    Le modèle est construit paresseusement au premier `encode()` afin
    d'éviter tout import lourd (torch/sentence-transformers) pendant les
    tests unitaires qui injectent un `model` factice via le constructeur.
    """

    def __init__(self, model=None, model_name: str = DEFAULT_MODEL_NAME):
        self._model = model
        self._model_name = model_name

    def _model_or_build(self):
        if self._model is None:
            self._model = _build_model(self._model_name)
        return self._model

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> EncodeResult:
        if not texts:
            return EncodeResult(dense=[], sparse=[], colbert=[], lexical_weights=[])
        m = self._model_or_build()
        dense_raw = m.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dense = [list(map(float, v)) for v in dense_raw]
        return EncodeResult(dense=dense, sparse=[], colbert=[], lexical_weights=[])


def _build_model(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": dtype})
